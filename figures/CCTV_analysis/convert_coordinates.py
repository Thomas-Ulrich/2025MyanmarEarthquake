from pyproj import Transformer

lon, lat = 96.03527777777778, 20.882055555555556


myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.92 +lat_0=22.00"
transformer = Transformer.from_crs("epsg:4326", myproj, always_xy=True)


x, y = transformer.transform(lon, lat)
print(x, y)
