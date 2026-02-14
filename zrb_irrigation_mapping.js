// ============================================================================
// Kafue Flats Irrigation Mapping - Google Earth Engine App
// ============================================================================
// Interactive viewer for Kafue Flats crop classification (2019-2024)
//
// Features:
// - Full-screen map with all classification layers (2019-2024) and reference datasets
// - Click anywhere on map to view NDVI/NDMI/NDWI time series (2019-2024)
// - Draw rectangle tool to calculate class areas across all years
// - Real-time stacked bar charts showing temporal area changes
// - Bottom panel: Time Series | Area Calculator | Legend
// ============================================================================

// ============================================================================
// CONFIGURATION
// ============================================================================

var CONFIG = {
  // Study area
  studyArea: ee.Geometry.Polygon([[
    [25.29357450684133, -17.124202489229475],
    [28.77624052246633, -17.124202489229475],
    [28.77624052246633, -14.699658421068504],
    [25.29357450684133, -14.699658421068504],
    [25.29357450684133, -17.124202489229475]
  ]]),
  center: {lon: 28.048, lat: -15.891, zoom: 10},

  // Years to display
  years: [2019, 2020, 2021, 2022, 2023, 2024],

  // External links
  githubRepoUrl: 'https://github.com/wyattarnold/ZRB-Irrigation-Mapping.git',

  // Asset paths
  assets: {
    classificationBase: 'projects/ee-warnold/assets/kafue_flats/classification/classified_',
    consolidatedEarly: 'projects/ee-warnold/assets/kafue_flats/classification/consolidated_2024_weighted',
    consolidatedLate: 'projects/ee-warnold/assets/kafue_flats/classification/consolidated_2019_weighted',
    consolidatedCombined: 'projects/ee-warnold/assets/kafue_flats/classification/consolidated_all_years_classified',
    dwLandcoverBase: 'projects/ee-warnold/assets/classification/dw_landcover_',
    trainingBase: 'projects/ee-warnold/assets/kafue_flats/training/training_samples_',
    hccp: 'projects/ee-warnold/assets/crop_masks/HCCP',
    budkyoRainfed: 'projects/ee-warnold/assets/crop_masks/Budkyo_Rainfed',
    budkyoIrrigated: 'projects/ee-warnold/assets/crop_masks/Budkyo_Irrigated'
  },

  // Sentinel-2 settings
  sentinel2: {
    collection: 'COPERNICUS/S2_SR_HARMONIZED',
    cloudScore: 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED',
    cloudThreshold: 0.60,
    drySeasonMonths: [5, 6, 7, 8, 9]  // May-September
  },

  // Dynamic World landcover palette (classes 0-8)
  dwPalette: ['419BDF', '397D49', '88B053', '7A87C6', 'E49635', 'DFC35A', 'C4281B', 'A59B8F', 'B39FE1'],
  dwLabels: ['Water', 'Trees', 'Grass', 'Flooded Veg.', 'Crops', 'Shrubs', 'Urban', 'Bare', 'Snow'],

  // Colors for aggregated class types (for charts)
  typeColors: {
    'Irrigated': '#7129A2',      // Purple (mid-tone from irrigated gradient)
    'Rainfed': '#f4912e',        // Orange
    'Urban': '#000000',          // Black
    'Water': '#0077be',          // Blue
    'Native Veg.': '#228b22',    // Forest green
    'Bare': '#a59b8f'            // Tan/brown
  },

  // Classification scheme (from __config__.py)
  classes: {
    // Crop clusters 0-4
    0: {name: 'Rainfed (0)', color: '#f4912e', type: 'Rainfed'},
    1: {name: 'Rainfed (1)', color: '#bd5e00', type: 'Rainfed'},
    2: {name: 'Irrigated (2)', color: '#4B0974', type: 'Irrigated'},
    3: {name: 'Irrigated (3)', color: '#67179C', type: 'Irrigated'},
    4: {name: 'Irrigated (4)', color: '#842FBD', type: 'Irrigated'},
    // Landcover classes 5-11
    5: {name: 'Urban', color: '#000000', type: 'Urban'},
    6: {name: 'Water', color: '#0055be', type: 'Water'},
    7: {name: 'Flooded Veg.', color: '#00a86b', type: 'Native Veg.'},
    8: {name: 'Trees', color: '#1a6e1a', type: 'Native Veg.'},
    9: {name: 'Shrubs', color: '#beda20', type: 'Native Veg.'},
    10: {name: 'Grass', color: '#90ee90', type: 'Native Veg.'},
    11: {name: 'Bare', color: '#a59b8f', type: 'Bare'}
  },

  // Consolidated classes (aggregated outputs)
  consolidatedClasses: {
    0: {name: 'Rainfed', color: '#f4912e'},
    1: {name: 'Irrigated', color: '#67179C'},
    2: {name: 'Urban', color: '#000000'},
    3: {name: 'Water', color: '#0055be'},
    4: {name: 'Flooded Veg.', color: '#00a86b'},
    5: {name: 'Trees', color: '#1a6e1a'},
    6: {name: 'Shrubs', color: '#beda20'},
    7: {name: 'Grass', color: '#90ee90'},
    8: {name: 'Bare', color: '#a59b8f'}
  }
};

// Generate classification palette for visualization
// The asset pixel values are sequential 0-11:
//   Values 0-4: Crop classes (rainfed 0-1, irrigated 2-4)
//   Values 5-11: Landcover classes (urban, water, flooded veg, trees, shrubs, grass, bare)
// GEE palettes should NOT have # prefix
CONFIG.classPalette = [
  CONFIG.classes[0].color.replace('#', ''),   // pixel value 0 → class 0: Rainfed (0)
  CONFIG.classes[1].color.replace('#', ''),   // pixel value 1 → class 1: Rainfed (1)
  CONFIG.classes[2].color.replace('#', ''),   // pixel value 2 → class 2: Irrigated (2)
  CONFIG.classes[3].color.replace('#', ''),   // pixel value 3 → class 3: Irrigated (3)
  CONFIG.classes[4].color.replace('#', ''),   // pixel value 4 → class 4: Irrigated (4)
  CONFIG.classes[5].color.replace('#', ''),   // pixel value 5 → class 5: Urban
  CONFIG.classes[6].color.replace('#', ''),   // pixel value 6 → class 6: Water
  CONFIG.classes[7].color.replace('#', ''),   // pixel value 7 → class 7: Flooded Veg
  CONFIG.classes[8].color.replace('#', ''),   // pixel value 8 → class 8: Trees
  CONFIG.classes[9].color.replace('#', ''),   // pixel value 9 → class 9: Shrubs
  CONFIG.classes[10].color.replace('#', ''),  // pixel value 10 → class 10: Grass
  CONFIG.classes[11].color.replace('#', '')   // pixel value 11 → class 11: Bare
];

// Map pixel values (0-11) to class IDs (direct mapping now)
// This is needed for area statistics to properly group by class type
CONFIG.pixelToClassId = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

// Consolidated class palette (no # prefix)
CONFIG.consolidatedPalette = [
  CONFIG.consolidatedClasses[0].color.replace('#', ''),
  CONFIG.consolidatedClasses[1].color.replace('#', ''),
  CONFIG.consolidatedClasses[2].color.replace('#', ''),
  CONFIG.consolidatedClasses[3].color.replace('#', ''),
  CONFIG.consolidatedClasses[4].color.replace('#', ''),
  CONFIG.consolidatedClasses[5].color.replace('#', ''),
  CONFIG.consolidatedClasses[6].color.replace('#', ''),
  CONFIG.consolidatedClasses[7].color.replace('#', ''),
  CONFIG.consolidatedClasses[8].color.replace('#', '')
];

// Brighter consolidated palette for probability hillshade rendering
CONFIG.consolidatedPaletteBright = [
  'ffb15e', // Rainfed
  '8d59cc', // Irrigated
  '000000', // Urban
  '2f8dff', // Water
  '25c79a', // Flooded Veg.
  '39a14a', // Trees
  'd4e545', // Shrubs
  'b9fcb9', // Grass
  'b9afa3'  // Bare
];

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Cloud masking for Sentinel-2
function maskClouds(image) {
  var cloudScore = ee.Image(image.get('cloud_score'));
  var clearSky = cloudScore.select('cs').gte(CONFIG.sentinel2.cloudThreshold);
  return image.updateMask(clearSky);
}

// Get dry season composite for a year
function getDrySeasonComposite(year) {
  var startDate = ee.Date.fromYMD(year, 5, 1);
  var endDate = ee.Date.fromYMD(year, 9, 30);

  var s2 = ee.ImageCollection(CONFIG.sentinel2.collection)
    .filterBounds(CONFIG.studyArea)
    .filterDate(startDate, endDate)
    .select(['B2', 'B3', 'B4', 'B8']);

  var cloudScores = ee.ImageCollection(CONFIG.sentinel2.cloudScore)
    .filterBounds(CONFIG.studyArea)
    .filterDate(startDate, endDate);

  var joined = ee.ImageCollection(ee.Join.saveFirst('cloud_score').apply({
    primary: s2,
    secondary: cloudScores,
    condition: ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
  }));

  var masked = joined.map(maskClouds);
  var composite = masked.median();

  return composite;
}

// Calculate vegetation indices
function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI');
  return image.addBands([ndvi, ndwi, ndmi]);
}

// Get time series for a point
function getTimeSeries(point, yearStart, yearEnd) {
  var startDate = ee.Date.fromYMD(yearStart, 1, 1);
  var endDate = ee.Date.fromYMD(yearEnd, 12, 31);

  var s2 = ee.ImageCollection(CONFIG.sentinel2.collection)
    .filterBounds(point)
    .filterDate(startDate, endDate)
    .select(['B2', 'B3', 'B4', 'B8', 'B11']);

  var cloudScores = ee.ImageCollection(CONFIG.sentinel2.cloudScore)
    .filterBounds(point)
    .filterDate(startDate, endDate);

  var joined = ee.ImageCollection(ee.Join.saveFirst('cloud_score').apply({
    primary: s2,
    secondary: cloudScores,
    condition: ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
  }));

  var masked = joined.map(maskClouds).map(addIndices);

  return masked;
}

// Create hillshaded visualization for consolidated assets
function createConsolidatedHillshade(image) {
  var classImage = image.select([0]);  // Band 0: winning class
  var probImage = image.select([1]);   // Band 1: probability (0-100)

  // RGB visualization of classes on [0, 1]
  var classRgb = classImage
    .visualize({min: 0, max: 8, palette: CONFIG.consolidatedPaletteBright})
    .divide(255);

  // Hillshade from probability (0-100)
  var probHillshade = ee.Terrain.hillshade(probImage).divide(255);

  // Keep confidence influence while preserving readability (darker)
  var brightness = probHillshade.multiply(0.55).add(0.30);
  var boostedRgb = classRgb.multiply(0.75).add(0.08);

  // Combine
  return boostedRgb.multiply(brightness);
}

// ============================================================================
// UI SETUP
// ============================================================================

// Clear UI
ui.root.clear();

// Create map panel
var mapPanel = ui.Map();
mapPanel.centerObject(CONFIG.studyArea);
mapPanel.setZoom(CONFIG.center.zoom + 1);
mapPanel.setOptions('SATELLITE');
mapPanel.setControlVisibility({layerList: true});

// Info button linking to GitHub repo
var infoButton = ui.Label('ℹ Info / GitHub', {
  margin: '1px 0px 1px 6px',
  padding: '5px 8px',
  backgroundColor: 'rgba(255,255,255,0.9)',
  border: '1px solid #999',
  fontSize: '11px',
  fontWeight: 'bold',
  color: '#1a73e8'
}, CONFIG.githubRepoUrl);

// ============================================================================
// INITIALIZE ALL LAYERS
// ============================================================================

// Add consolidated layers (early-weighted visible by default)
var consolidatedEarly = ee.Image(CONFIG.assets.consolidatedEarly);
var consolidatedLate = ee.Image(CONFIG.assets.consolidatedLate);
var consolidatedCombined = ee.Image(CONFIG.assets.consolidatedCombined);

mapPanel.addLayer(
  createConsolidatedHillshade(consolidatedEarly),
  {},
  'Consolidated (2024-weighted)',
  true
);

mapPanel.addLayer(
  createConsolidatedHillshade(consolidatedLate),
  {},
  'Consolidated (2019-weighted)',
  false
);

mapPanel.addLayer(
  createConsolidatedHillshade(consolidatedCombined),
  {},
  'Consolidated (All-years combined)',
  false
);

// Add classification layers for all years (only 2024 visible by default)
CONFIG.years.forEach(function(year) {
  var classified = ee.Image(CONFIG.assets.classificationBase + year);
  var classImage = classified.select([0]);

  var isVisible = false;
  mapPanel.addLayer(classImage, {
    min: 0,
    max: 11,
    palette: CONFIG.classPalette
  }, 'Classification ' + year, isVisible);
});

// Add Sentinel-2 true color composites for all years (unchecked by default)
CONFIG.years.forEach(function(year) {
  var composite = getDrySeasonComposite(year);
  var trueColor = composite.select(['B4', 'B3', 'B2']);
  mapPanel.addLayer(trueColor, {
    min: 0,
    max: 3000,
    gamma: 1.4
  }, 'Sentinel-2 ' + year, false);
});

// Add DW landcover layers for all years (unchecked by default)
CONFIG.years.forEach(function(year) {
  var dwLandcover = ee.Image(CONFIG.assets.dwLandcoverBase + year);
  mapPanel.addLayer(dwLandcover, {
    min: 0,
    max: 8,
    palette: CONFIG.dwPalette
  }, 'DW Landcover ' + year, false);
});

// Add crop mask reference layers (unchecked by default)
var hccp = ee.Image(CONFIG.assets.hccp);
var hccpMasked = hccp.updateMask(hccp.gt(0.0));
var hccpViz = {min: 0, max: 100, palette: ['00FFFF', '0000FF'], opacity: 0.7};
mapPanel.addLayer(hccpMasked, hccpViz, 'HCCP', false);

var bkrain = ee.Image(CONFIG.assets.budkyoRainfed);
var bkrainMasked = bkrain.updateMask(bkrain.gt(0.0));
var bkrainViz = {min: 0, max: 100, palette: ['DFD7A7', 'DFD7A7'], opacity: 0.7};
mapPanel.addLayer(bkrainMasked, bkrainViz, 'Budkyo Rainfed', false);

var bkirr = ee.Image(CONFIG.assets.budkyoIrrigated);
var bkirrMasked = bkirr.updateMask(bkirr.gt(0.0));
var bkirrViz = {min: 0, max: 100, palette: ['00FFFF', '0000FF'], opacity: 0.7};
mapPanel.addLayer(bkirrMasked, bkirrViz, 'Budkyo Irrigated', false);

// Shared drawing controls (outside charts)
var drawControlPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    padding: '4px 8px',
    backgroundColor: '#f7f7f7'
  }
});

var appTitle = ui.Label('Zambezi Irrigation Mapping', {
  fontSize: '13px',
  fontWeight: 'bold',
  color: '#222',
  margin: '4px 10px 4px 0px'
});

var drawRectButton = ui.Button({
  label: '▭ Rectangle',
  onClick: function() {
    chartPanel.clear();
    chartPanel.add(ui.Label('Draw a rectangle on the map...', {color: '#666'}));
    areaChartPanel.clear();
    areaChartPanel.add(ui.Label('Draw a rectangle on the map...', {color: '#666'}));
    consolidatedChartPanel.clear();
    consolidatedChartPanel.add(ui.Label('Draw a rectangle on the map...', {color: '#666'}));
    mapPanel.drawingTools().setShape('rectangle');
    mapPanel.drawingTools().draw();
  },
  style: {margin: '1px', fontSize: '10px'}
});

var drawPolyButton = ui.Button({
  label: '⬡ Polygon',
  onClick: function() {
    chartPanel.clear();
    chartPanel.add(ui.Label('Draw a polygon on the map (double-click to finish)...', {color: '#666'}));
    areaChartPanel.clear();
    areaChartPanel.add(ui.Label('Draw a polygon on the map (double-click to finish)...', {color: '#666'}));
    consolidatedChartPanel.clear();
    consolidatedChartPanel.add(ui.Label('Draw a polygon on the map (double-click to finish)...', {color: '#666'}));
    mapPanel.drawingTools().setShape('polygon');
    mapPanel.drawingTools().draw();
  },
  style: {margin: '1px', fontSize: '10px'}
});

var leftControlPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    stretch: 'horizontal'
  }
});

leftControlPanel.add(appTitle);
var drawToolsCenterPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    stretch: 'horizontal'
  }
});

drawToolsCenterPanel.add(ui.Label('', {stretch: 'horizontal'}));
drawToolsCenterPanel.add(drawRectButton);
drawToolsCenterPanel.add(drawPolyButton);
drawToolsCenterPanel.add(ui.Label('', {stretch: 'horizontal'}));

var rightControlPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    stretch: 'horizontal'
  }
});

rightControlPanel.add(ui.Label('', {stretch: 'horizontal'}));
rightControlPanel.add(infoButton);

drawControlPanel.add(leftControlPanel);
drawControlPanel.add(drawToolsCenterPanel);
drawControlPanel.add(rightControlPanel);


// ============================================================================
// BOTTOM PANEL (3 SECTIONS: TIME SERIES | AREA CALCULATOR | LEGEND)
// ============================================================================

var bottomPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    height: '330px',
    padding: '10px',
    backgroundColor: '#f0f0f0'
  }
});

// Time series panel (left)
var timeSeriesPanel = ui.Panel({
  style: {
    width: '40%',
    padding: '5px',
    backgroundColor: 'white',
    margin: '0px 5px 0px 0px'
  }
});

var timeSeriesTitle = ui.Label({
  value: 'Time Series Inspector',
  style: {fontWeight: 'bold', margin: '5px'}
});
timeSeriesPanel.add(timeSeriesTitle);

var timeSeriesInfo = ui.Label('Draw a shape to view average NDVI, NDMI, NDWI (2019-2024)', {
  fontSize: '11px',
  color: '#666',
  margin: '5px'
});
timeSeriesPanel.add(timeSeriesInfo);

var chartPanel = ui.Panel({
  style: {height: '255px'}
});
timeSeriesPanel.add(chartPanel);

// Area calculator panel (middle)
var areaPanel = ui.Panel({
  style: {
    width: '40%',
    padding: '5px',
    backgroundColor: 'white',
    margin: '0px 5px'
  }
});

var areaTitle = ui.Label({
  value: 'Area Calculator',
  style: {fontWeight: 'bold', margin: '5px'}
});
areaPanel.add(areaTitle);

var areaInfo = ui.Label('Draw a shape to calculate class areas', {
  fontSize: '11px',
  color: '#666',
  margin: '5px'
});
areaPanel.add(areaInfo);

var areaChartPanel = ui.Panel({
  style: {height: '255px'}
});
areaPanel.add(areaChartPanel);

// Consolidated area panel (new)
var consolidatedPanel = ui.Panel({
  style: {
    width: '12%',
    padding: '5px',
    backgroundColor: 'white',
    margin: '0px 5px'
  }
});

var consolidatedTitle = ui.Label({
  value: 'Consolidated Area',
  style: {fontWeight: 'bold', margin: '5px'}
});
consolidatedPanel.add(consolidatedTitle);

var consolidatedInfo = ui.Label('Stacked area by class type from all-years consolidated map', {
  fontSize: '11px',
  color: '#666',
  margin: '5px'
});
consolidatedPanel.add(consolidatedInfo);

var consolidatedChartPanel = ui.Panel({
  style: {height: '255px'}
});
consolidatedPanel.add(consolidatedChartPanel);

// Legend panel (right) - narrower width
var legendOuterPanel = ui.Panel({
  style: {
    width: '8%',
    padding: '5px',
    backgroundColor: 'white',
    margin: '0px 0px 0px 5px'
  }
});

var legendTitle = ui.Label({
  value: 'Legend',
  style: {fontWeight: 'bold', margin: '3px', fontSize: '10px'}
});
legendOuterPanel.add(legendTitle);

var legendPanel = ui.Panel({
  style: {
    padding: '2px',
    maxHeight: '270px'
  }
});

// Consolidated legend (default)
var consolidatedClassIds = [0, 1, 2, 3, 4, 5, 6, 7, 8];

function addConsolidatedLegend(classIds) {
  var sectionLabel = ui.Label('Consolidated Classes:', {
    fontWeight: 'bold',
    fontSize: '8px',
    margin: '3px 0px 1px 0px'
  });
  legendPanel.add(sectionLabel);

  classIds.forEach(function(classId) {
    var classInfo = CONFIG.consolidatedClasses[classId];
    var colorBox = ui.Label({
      style: {
        backgroundColor: classInfo.color,
        padding: '4px',
        margin: '1px 3px 1px 0px',
        border: '1px solid black'
      }
    });

    var description = ui.Label(classInfo.name, {
      margin: '1px 0px',
      fontSize: '8px'
    });

    var row = ui.Panel({
      widgets: [colorBox, description],
      layout: ui.Panel.Layout.Flow('horizontal')
    });

    legendPanel.add(row);
  });
}

addConsolidatedLegend(consolidatedClassIds);

legendOuterPanel.add(legendPanel);

bottomPanel.add(timeSeriesPanel);
bottomPanel.add(areaPanel);
bottomPanel.add(consolidatedPanel);
bottomPanel.add(legendOuterPanel);

// ============================================================================
// INTERACTIVE TOOLS
// ============================================================================

// Drawing tools handler
var drawingTools = mapPanel.drawingTools();
drawingTools.setShown(false);

drawingTools.onDraw(function() {
  var layers = drawingTools.layers();
  if (layers.length() === 0) return;

  var geometry = layers.get(0).toGeometry();

  // Shared geometry for both analyses
  processTimeSeriesDrawing(geometry);
  processAreaDrawing(geometry);
  
  // Clear drawing after calculation
  drawingTools.layers().reset();
});

// Process time series for drawn region
function processTimeSeriesDrawing(geometry) {
  chartPanel.clear();
  chartPanel.add(ui.Label('Loading time series (this may take a moment)...', {color: '#666'}));

  var timeSeries = getTimeSeries(geometry, 2019, 2024);

  // Create chart with region average
  var chart = ui.Chart.image.series({
    imageCollection: timeSeries.select(['NDVI', 'NDMI', 'NDWI']),
    region: geometry,
    reducer: ee.Reducer.mean(),
    scale: 30  // Coarser scale for faster computation over areas
  }).setOptions({
    title: 'Vegetation Indices Time Series (Area Average)',
    vAxis: {title: 'Index Value', viewWindow: {min: -1, max: 1}},
    hAxis: {title: 'Date'},
    lineWidth: 1,
    pointSize: 2,
    series: {
      0: {color: 'green', labelInLegend: 'NDVI'},
      1: {color: 'blue', labelInLegend: 'NDMI'},
      2: {color: 'cyan', labelInLegend: 'NDWI'}
    }
  });

  chartPanel.clear();
  chartPanel.add(chart);
}

// Process area statistics for drawn region
function processAreaDrawing(geometry) {
  areaChartPanel.clear();
  areaChartPanel.add(ui.Label('Calculating areas for all years...', {color: '#666'}));
  consolidatedChartPanel.clear();
  consolidatedChartPanel.add(ui.Label('Calculating all-years consolidated area...', {color: '#666'}));

  // Calculate areas for all years
  var allYearResults = {};
  var yearsCompleted = 0;

  CONFIG.years.forEach(function(year) {
    var classified = ee.Image(CONFIG.assets.classificationBase + year);
    var pixelArea = ee.Image.pixelArea();
    var classImage = classified.select([0]);
    var areaImage = pixelArea.addBands(classImage);

    var areas = areaImage.reduceRegion({
      reducer: ee.Reducer.sum().group({
        groupField: 1,
        groupName: 'class'
      }),
      geometry: geometry,
      scale: 10,
      maxPixels: 1e9,
      bestEffort: true
    });

    areas.get('groups').evaluate(function(groups) {
      allYearResults[year] = groups || [];
      yearsCompleted++;

      // Once all years are done, display results
      if (yearsCompleted === CONFIG.years.length) {
        displayAllYearResults(allYearResults);
      }
    });
  });

  function displayAllYearResults(results) {
    areaChartPanel.clear();

    // Aggregate data across all years for chart
    var areasByTypeAndYear = {};
    var totalAreaByYear = {};

    CONFIG.years.forEach(function(year) {
      var groups = results[year];
      var yearTotal = 0;

      if (!areasByTypeAndYear[year]) {
        areasByTypeAndYear[year] = {};
      }

      groups.forEach(function(group) {
        var pixelValue = group['class'];  // This is 0-14 from the asset
        var areaM2 = group.sum;
        var areaHa = areaM2 / 10000;
        yearTotal += areaHa;

        // Map pixel value back to original class ID
        var classId = CONFIG.pixelToClassId[pixelValue];
        if (CONFIG.classes[classId]) {
          var type = CONFIG.classes[classId].type;
          if (!areasByTypeAndYear[year][type]) {
            areasByTypeAndYear[year][type] = 0;
          }
          areasByTypeAndYear[year][type] += areaHa;
        }
      });

      totalAreaByYear[year] = yearTotal;
    });

    // Create stacked bar chart data
    // Get all unique class types in a consistent order
    var typesOrder = ['Irrigated', 'Rainfed', 'Native Veg.', 'Water', 'Urban', 'Bare'];
    var typesList = [];
    typesOrder.forEach(function(type) {
      var hasData = false;
      CONFIG.years.forEach(function(year) {
        if (areasByTypeAndYear[year][type]) {
          hasData = true;
        }
      });
      if (hasData) {
        typesList.push(type);
      }
    });

    // Build data table for stacked chart
    var chartData = [['Year'].concat(typesList)];
    CONFIG.years.forEach(function(year) {
      var row = [String(year)];
      typesList.forEach(function(type) {
        row.push(areasByTypeAndYear[year][type] || 0);
      });
      chartData.push(row);
    });

    // Build colors array matching the order of types
    var chartColors = [];
    typesList.forEach(function(type) {
      chartColors.push(CONFIG.typeColors[type] || '#666666');
    });

    // Create stacked bar chart
    var chart = ui.Chart(chartData, 'Table')
      .setChartType('ColumnChart')
      .setOptions({
        title: 'Area by Class Type (All Years)',
        vAxis: {title: 'Area (hectares)'},
        hAxis: {title: 'Year'},
        isStacked: true,
        legend: {position: 'right'},
        colors: chartColors,
        chartArea: {width: '60%', height: '60%'}
      });

    areaChartPanel.add(chart);

    // Stacked bar for all-years consolidated classification only
    var consolidatedTypeMap = {
      0: 'Rainfed',
      1: 'Irrigated',
      2: 'Urban',
      3: 'Water',
      4: 'Native Veg.',
      5: 'Native Veg.',
      6: 'Native Veg.',
      7: 'Native Veg.',
      8: 'Bare'
    };

    var consolidatedAreaImage = ee.Image.pixelArea().addBands(consolidatedCombined.select([0]).rename('class'));
    var consolidatedAreas = consolidatedAreaImage.reduceRegion({
      reducer: ee.Reducer.sum().group({
        groupField: 1,
        groupName: 'class'
      }),
      geometry: geometry,
      scale: 10,
      maxPixels: 1e9,
      bestEffort: true
    });

    consolidatedAreas.get('groups').evaluate(function(groups) {
      var consolidatedByType = {};
      (groups || []).forEach(function(group) {
        var classId = group['class'];
        var type = consolidatedTypeMap[classId];
        if (!type) return;
        if (!consolidatedByType[type]) {
          consolidatedByType[type] = 0;
        }
        consolidatedByType[type] += group.sum / 10000;
      });

      var consolidatedTypesOrder = ['Irrigated', 'Rainfed', 'Native Veg.', 'Water', 'Urban', 'Bare'];
      var consolidatedTypes = [];
      consolidatedTypesOrder.forEach(function(type) {
        if (consolidatedByType[type]) {
          consolidatedTypes.push(type);
        }
      });

      var consolidatedChartData = [['All-years'].concat(consolidatedTypes)];
      var consolidatedRow = ['Consolidated'];
      consolidatedTypes.forEach(function(type) {
        consolidatedRow.push(consolidatedByType[type] || 0);
      });
      consolidatedChartData.push(consolidatedRow);

      var consolidatedColors = [];
      consolidatedTypes.forEach(function(type) {
        consolidatedColors.push(CONFIG.typeColors[type] || '#666666');
      });

      var consolidatedChart = ui.Chart(consolidatedChartData, 'Table')
        .setChartType('ColumnChart')
        .setOptions({
          title: 'Area by Class Type (All-years Consolidated)',
          vAxis: {title: 'Area (hectares)'},
          hAxis: {title: 'Classification'},
          isStacked: true,
          legend: {position: 'right'},
          colors: consolidatedColors,
          chartArea: {width: '60%', height: '60%'}
        });

      consolidatedChartPanel.clear();
      consolidatedChartPanel.add(consolidatedChart);
    });
  }
}

// ============================================================================
// INITIALIZE
// ============================================================================

// Create main layout with map and bottom panel
var mainPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('vertical'),
  style: {stretch: 'both'}
});

mainPanel.add(mapPanel);
mainPanel.add(drawControlPanel);
mainPanel.add(bottomPanel);

// Add to UI root
ui.root.clear();
ui.root.add(mainPanel);

// Add study area boundary
mapPanel.addLayer(CONFIG.studyArea, {color: 'white'}, 'Study Area', false);

print('=== Kafue Flats Irrigation Mapping App ===');
print('Status: Initialized successfully');
print('');
print('Layers Added:');
print('  - Consolidated (2024-weighted)');
print('  - Consolidated (2019-weighted)');
print('  - Consolidated (All-years combined)');
print('  - Classification 2019-2024 (2024 visible by default)');
print('  - Sentinel-2 True Color 2019-2024 (all hidden by default)');
print('  - DW Landcover 2019-2024 (all hidden by default)');
print('  - HCCP (hidden by default)');
print('  - Budkyo Rainfed (hidden by default)');
print('  - Budkyo Irrigated (hidden by default)');
print('  - Study Area boundary (hidden by default)');
print('');
print('Controls:');
print('  - Layer Toggle: Use Layers panel (map top right)');
print('  - Shared Draw Controls: Use rectangle/polygon buttons above charts');
print('  - Legend: View at bottom right');
print('');
print('Tip: Toggle layers on/off using the Layers panel');
print('Info: Use the right-aligned "Info / GitHub" link in the draw controls bar.');
