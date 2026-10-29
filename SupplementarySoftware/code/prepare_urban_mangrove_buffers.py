
from pathlib import Path
import argparse,json
import geopandas as gpd
import numpy as np
import pyogrio,shapely
from pyproj import Transformer
from shapely.geometry import shape

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gub',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),'Choose a new output; existing files are not overwritten'
    features=json.loads(a.catalog.read_text('utf-8'))['features']
    assert len({f['properties']['tilename'] for f in features})==len(features)==1696
    to=Transformer.from_crs(4326,6933,always_xy=True)
    project=lambda geom:shapely.transform(geom,lambda xy:np.column_stack(to.transform(xy[:,0],xy[:,1])))
    tiles=project(np.array([shape(f['geometry']) for f in features],dtype=object))
    g=pyogrio.read_dataframe(a.gub,columns=['ORIG_FID','urbanArea'],where='urbanArea > 100')
    assert str(g.crs)=='EPSG:4326' and len(g)==1029
    buffers=shapely.buffer(shapely.simplify(project(g.geometry.to_numpy()),30,preserve_topology=True),2000,quad_segs=2)
    assert shapely.is_valid(buffers).all()
    ix=shapely.STRtree(tiles).query(buffers,predicate='intersects');keep=np.unique(ix[0]);assert len(keep)==152
    selected=g.iloc[keep].copy()
    output=gpd.GeoDataFrame(selected.drop(columns='geometry'),geometry=buffers[keep],crs='EPSG:6933')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pyogrio.write_dataframe(output,a.output,layer='gmw_intersecting_city_buffers',driver='GPKG')
    print(json.dumps({'status':'PASS','global_selected_cities':len(g),'candidate_buffers':len(output),'crs':'EPSG:6933',
      'simplification_m':30,'buffer_m':2000,'quad_segs':2,'includes_city_interiors':True}))
if __name__=='__main__':main()
