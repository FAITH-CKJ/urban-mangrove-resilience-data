







from __future__ import annotations
import argparse,datetime,hashlib,json,os,tarfile,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine,array_bounds
from rasterio.warp import reproject,Resampling
import shapely
from shapely.geometry import box
from pyproj import Transformer

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(d):print(json.dumps(dict(utc=now(),**d)),flush=True)
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def obtain_tiles(protocol,archive,tile_dirs,output):
    wanted={r['filename']:r for r in protocol['tiles']};found={}
    for name,rec in wanted.items():
        for folder in tile_dirs:
            p=folder/name
            if p.is_file():
                assert p.stat().st_size==rec['expected_uncompressed_bytes'] and sha(p)==rec['expected_sha256'],p
                found[name]=p;break
    missing=set(wanted)-set(found)
    if missing:
        assert archive and archive.is_file(),'Official annual archive required for missing global tiles'
        dest=output/'native_tiles';dest.mkdir(parents=True,exist_ok=True)
        for name in sorted(missing.copy()):
            p=dest/name;r=wanted[name]
            if p.is_file() and p.stat().st_size==r['expected_uncompressed_bytes'] and sha(p)==r['expected_sha256']:
                found[name]=p;missing.remove(name)
        if missing:
            with tarfile.open(archive,'r|gz') as tar:
                for member in tar:
                    name=Path(member.name).name
                    if name not in missing:continue
                    assert member.isfile() and not member.issym() and not member.islnk(),member.name
                    p=dest/name;temp=p.with_suffix('.partial.tif')
                    assert p.resolve().parent==dest.resolve()
                    stream=tar.extractfile(member);h=hashlib.sha256();size=0
                    with temp.open('wb') as f:
                        while True:
                            b=stream.read(8*1024*1024)
                            if not b:break
                            f.write(b);h.update(b);size+=len(b)
                    rec=wanted[name]
                    assert size==rec['expected_uncompressed_bytes'] and h.hexdigest()==rec['expected_sha256'],name
                    temp.replace(p);found[name]=p;missing.remove(name)
                    if len(found)%20==0:log({'stage':'extract','verified_tiles':len(found),'required_tiles':len(wanted)})
                    if not missing:break
    assert set(found)==set(wanted)
    dump(output/'native_tile_verification.json',{'status':'PASS','tile_count':len(found),
          'all_match_prior_product_checksums':True,'files':[{'file':n,'bytes':wanted[n]['expected_uncompressed_bytes'],
          'sha256':wanted[n]['expected_sha256']} for n in sorted(found)]})
    return found

def baseline_tile(rec,source,folder):
    dest=folder/rec['filename'];side=dest.with_suffix('.json')
    if dest.is_file() and side.is_file():
        d=json.loads(side.read_text('utf-8'))
        if d.get('source_sha256')==rec['expected_sha256'] and sha(dest)==d['sha256']:return d
    with rasterio.open(source) as src:
        assert src.count==41 and str(src.crs)=='EPSG:4326'
        assert [src.height,src.width]==rec['native_shape']
        assert np.allclose(tuple(src.transform)[:6],rec['transform'],rtol=0,atol=1e-12)
        a=src.read([16,17,18,19,20]);assert int(a.max())<=1
        union=np.any(a>0,axis=0).astype('uint8');del a
        tmp=dest.with_suffix('.partial.tif')
        with rasterio.open(tmp,'w',driver='GTiff',height=src.height,width=src.width,count=1,dtype='uint8',
             crs=src.crs,transform=src.transform,nodata=0,compress='DEFLATE',tiled=True,blockxsize=256,blockysize=256) as f:
            f.write(union,1);f.set_band_description(1,'GMW v4.1.12 union of annual presence 2000-2004')
        count=int(union.sum())
    tmp.replace(dest)
    d={'file':dest.name,'source_sha256':rec['expected_sha256'],'sha256':sha(dest),'native_baseline_pixels':count,
       'years':[2000,2001,2002,2003,2004],'bands':[16,17,18,19,20]}
    dump(side,d);return d

def worker_init(buffers_path,reference_path,tiles,baseline_dir,out,fingerprint):
    global BUFFERS,BTREE,TILES,TTREE,BASELINE,OUT,FINGERPRINT,REFS
    BUFFERS=pyogrio.read_dataframe(buffers_path).geometry.to_numpy();BTREE=shapely.STRtree(BUFFERS)
    to=Transformer.from_crs(4326,6933,always_xy=True);geoms=[]
    TILES=tiles
    for item in tiles:
        x0,y0,x1,y1=item['bounds'];a,b=to.transform(x0,y0);c,d=to.transform(x1,y1);geoms.append(box(a,b,c,d))
    TTREE=shapely.STRtree(geoms);BASELINE=Path(baseline_dir);OUT=Path(out);FINGERPRINT=fingerprint
    with np.load(reference_path) as z:
        x=z['grid_x'];y=z['grid_y'];reg=z['region_id']
    keys=np.stack([x//1024,y//1024],axis=1);REFS={}
    for pair in np.unique(keys,axis=0):
        m=(keys==pair).all(axis=1)
        REFS[tuple(pair.tolist())]=(1023-y[m]%1024,x[m]%1024,reg[m])

def calculate_block(pair):
    bx,by=pair;key=f'x{bx}_y{by}';side=OUT/'blocks'/(key+'.json');npz=OUT/'blocks'/(key+'.npz')
    if side.is_file() and npz.is_file():
        old=json.loads(side.read_text('utf-8'))
        if old.get('fingerprint')==FINGERPRINT and sha(npz)==old['mask_sha256']:return old
    transform=Affine(30,0,bx*30720-960,0,-30,(by+1)*30720+960);shape=(1088,1088)
    bounds=box(*array_bounds(*shape,transform));ui=BTREE.query(bounds,predicate='intersects')
    urban=rasterize(((BUFFERS[i],1) for i in ui),out_shape=shape,transform=transform,fill=0,dtype='uint8',all_touched=False) if len(ui) else np.zeros(shape,'uint8')
    baseline=np.zeros(shape,'uint8');tile_indices=TTREE.query(bounds,predicate='intersects')
    if urban.any():
        with rasterio.Env(GDAL_CACHEMAX=32*1024*1024):
            for i in sorted(tile_indices):
                with rasterio.open(BASELINE/TILES[i]['filename']) as src:
                    dst=np.zeros(shape,'uint8')
                    reproject(rasterio.band(src,1),dst,src_transform=src.transform,src_crs=src.crs,src_nodata=0,
                              dst_transform=transform,dst_crs='EPSG:6933',dst_nodata=0,resampling=Resampling.nearest,
                              num_threads=1,warp_mem_limit=64)
                    baseline|=dst
    core=(baseline[32:1056,32:1056]>0)&(urban[32:1056,32:1056]>0)
    n=int(core.sum());rows,cols,reg=REFS.get((bx,by),(np.array([],int),np.array([],int),np.array([],int)))
    ok=core[rows,cols];nr=len(rows);inside=int(ok.sum())
    miss=[{'grid_x':int(bx*1024+c),'grid_y':int((by+1)*1024-r-1),'region_id':int(g)} for r,c,g in zip(rows[~ok],cols[~ok],reg[~ok])]
    npz.parent.mkdir(parents=True,exist_ok=True);temp=npz.with_suffix('.partial.npz')
    np.savez_compressed(temp,eligibility=np.packbits(core,bitorder='little'),shape=np.array([1024,1024],np.uint16),block_xy=np.array([bx,by],np.int32))
    temp.replace(npz)
    result={'status':'COMPLETE','fingerprint':FINGERPRINT,'block':key,'block_x':bx,'block_y':by,
            'global_baseline_centres':n,'actual_reference_centres':nr,'reference_inside_global':inside,
            'reference_outside_global':nr-inside,'reference_missing_cells':miss,
            'reference_inside_by_region':np.bincount(reg[ok].astype(int),minlength=11)[1:].tolist(),
            'source_tiles':[TILES[i]['filename'] for i in sorted(tile_indices)],'mask_sha256':sha(npz)}
    dump(side,result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--buffers',type=Path,required=True)
    parser.add_argument('--reference-cells',type=Path,required=True);parser.add_argument('--gmw-archive',type=Path)
    parser.add_argument('--tile-dir',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(args.protocol.read_text('utf-8'));start=time.monotonic()
    assert protocol['years']==[2000,2001,2002,2003,2004] and protocol['grid']['resolution_m']==30
    input_identity={'protocol_sha256':sha(args.protocol),'buffers_sha256':sha(args.buffers),
       'reference_cells_sha256':sha(args.reference_cells),'producer_sha256':sha(Path(__file__))}
    fingerprint=hashlib.sha256(json.dumps(input_identity,sort_keys=True).encode()).hexdigest()
    dump(out/'run_state.json',{'status':'RUNNING','phase':'VERIFY_NATIVE_TILES','fingerprint':fingerprint,'started_utc':now()})
    sources=obtain_tiles(protocol,args.gmw_archive,args.tile_dir,out)
    baseline=out/'baseline_tiles';baseline.mkdir(exist_ok=True)
    records=[]
    with ThreadPoolExecutor(max_workers=min(args.workers,4)) as pool:
        jobs=[pool.submit(baseline_tile,r,sources[r['filename']],baseline) for r in protocol['tiles']]
        for f in as_completed(jobs):
            records.append(f.result())
            if len(records)%20==0:log({'stage':'baseline_union','completed_tiles':len(records),'total_tiles':len(jobs)})
    dump(out/'baseline_derivation.json',{'status':'PASS','records':sorted(records,key=lambda x:x['file'])})
    
    
    buffers=pyogrio.read_dataframe(args.buffers).geometry.to_numpy();to=Transformer.from_crs(4326,6933,always_xy=True)
    tg=[]
    for r in protocol['tiles']:
        x0,y0,x1,y1=r['bounds'];a,b=to.transform(x0,y0);c,d=to.transform(x1,y1);tg.append(box(a,b,c,d))
    tg=np.array(tg,dtype=object);ij=shapely.STRtree(tg).query(buffers,predicate='intersects')
    candidate=set()
    for x0,y0,x1,y1 in shapely.bounds(shapely.intersection(buffers[ij[0]],tg[ij[1]])):
        for x in range(int(np.floor(x0/30720)),int(np.floor(x1/30720))+1):
            for y in range(int(np.floor(y0/30720)),int(np.floor(y1/30720))+1):candidate.add((x,y))
    assert candidate==set(tuple(x) for x in protocol['blocks'])
    results=[]
    init=(str(args.buffers),str(args.reference_cells),protocol['tiles'],str(baseline),str(out),fingerprint)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=init) as pool:
        jobs={pool.submit(calculate_block,pair):pair for pair in sorted(candidate)}
        for future in as_completed(jobs):
            results.append(future.result())
            if len(results)%25==0 or len(results)==len(jobs):
                state={'status':'RUNNING','phase':'GLOBAL_GRID_AGGREGATION','fingerprint':fingerprint,
                       'completed_blocks':len(results),'total_blocks':len(jobs),'updated_utc':now(),
                       'elapsed_seconds':round(time.monotonic()-start,1)}
                dump(out/'run_state.json',state);log(state)
    results.sort(key=lambda x:(x['block_x'],x['block_y']))
    frame=pd.DataFrame([{k:r[k] for k in ['block','block_x','block_y','global_baseline_centres','actual_reference_centres','reference_inside_global','reference_outside_global']} for r in results])
    for name in ['global_baseline','actual_reference','reference_inside_global','reference_outside_global']:
        col=name+'_centres' if name in ['global_baseline','actual_reference'] else name
        frame[name+'_area_km2']=frame[col]*.0009
    frame.to_csv(out/'coverage_by_block.csv',index=False)
    denominator=int(frame.global_baseline_centres.sum());numerator=int(frame.reference_inside_global.sum());actual=int(frame.actual_reference_centres.sum())
    missing=[v for r in results for v in r['reference_missing_cells']]
    total_regions=np.sum([r['reference_inside_by_region'] for r in results],axis=0).tolist()
    summary={'status':'PASS' if not missing else 'REFERENCE_MEMBERSHIP_MISMATCH','identity':protocol['identity'],
        'completed_utc':now(),'fingerprint':fingerprint,'inputs':input_identity,
        'global_baseline_centres':denominator,'global_baseline_area_km2':denominator*.0009,
        'actual_reference_centres':actual,'actual_reference_area_km2':actual*.0009,
        'reference_inside_global_centres':numerator,'reference_inside_global_area_km2':numerator*.0009,
        'reference_outside_global_centres':len(missing),'coverage_fraction':numerator/denominator,
        'coverage_percent':100*numerator/denominator,'exceeds_three_quarters':numerator*4>denominator*3,
        'unrepresented_global_centres':denominator-numerator,'unrepresented_global_area_km2':(denominator-numerator)*.0009,
        'nonempty_global_blocks':int((frame.global_baseline_centres>0).sum()),'candidate_blocks':len(candidate),
        'reference_inside_by_region':total_regions,'missing_reference_cells':missing,
        'selected_global_city_count':protocol['selected_global_city_count'],'candidate_city_count':len(buffers),
        'official_global_tile_count':protocol['global_product_tiles'],'tiles_checked':len(sources),
        'interpretation':'Mapped 2000-2004 historical mangrove area in the fixed GUB2018 large-city interface; the same projected buffer, equal-area grid, centre rule and baseline years are used for both sides.',
        'elapsed_seconds':time.monotonic()-start}
    dump(out/'coverage_summary.json',summary)
    assert actual==protocol['reference_centres']==934814
    assert not missing, f'{len(missing)} actual reference centres outside recomputed global support'
    
    
    gx=[];gy=[]
    for result in results:
        if result['global_baseline_centres']==0:continue
        with np.load(out/'blocks'/(result['block']+'.npz')) as z:
            a=np.unpackbits(z['eligibility'],bitorder='little',count=1024*1024).reshape(1024,1024)
        rr,cc=np.nonzero(a);gx.append((result['block_x']*1024+cc).astype(np.int32));gy.append(((result['block_y']+1)*1024-1-rr).astype(np.int32))
    gx=np.concatenate(gx);gy=np.concatenate(gy);assert len(gx)==denominator
    np.savez_compressed(out/'global_baseline_cells.npz',grid_x=gx,grid_y=gy,pixel_area_m2=np.int32(900))
    dump(out/'run_state.json',{'status':'COMPLETE','phase':'VERIFIED_TERMINAL','fingerprint':fingerprint,
         'completed_utc':now(),'coverage_percent':summary['coverage_percent'],'global_centres':denominator,
         'reference_centres':numerator,'elapsed_seconds':time.monotonic()-start})
    log(summary)

if __name__=='__main__':main()
