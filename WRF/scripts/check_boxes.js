var dataset = ee.FeatureCollection('JRC/GWIS/GlobFire/v2/FinalPerimeters');

var fireId = 20777134;

var fire = dataset.filter(ee.Filter.eq('Id', fireId));
Map.addLayer(fire, {color: 'red'}, 'Fire ID: 20777134');

Map.centerObject(fire, 9);

var cenLat = 41.53681;
var cenLon = -123.5552;

var centerPoint = ee.Geometry.Point([cenLon, cenLat]);
var kmBox = centerPoint.buffer(150000).bounds();
Map.addLayer(kmBox, {color: 'blue'}, '300 km Box');

var centroid = fire.geometry().centroid();
Map.addLayer(centroid, {color: 'green'}, 'Fire Centroid');

var lon = centroid.coordinates().get(0);
var lat = centroid.coordinates().get(1);

var degBox = ee.Geometry.Rectangle([
  ee.Number(lon).subtract(0.5),
  ee.Number(lat).subtract(0.5),
  ee.Number(lon).add(0.5),
  ee.Number(lat).add(0.5)
]);

Map.addLayer(degBox, {color: 'purple'}, '1° Lat/Lon Box');
