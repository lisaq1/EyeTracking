def make_text_file(configPath, imagePath, eyeFocalDist, eyeFocalLength, pupilDiameter, imageDist, \
displayPreset, displayPixelWidth, displayPixelHeight, displayPPI, hardwareType, \
hardwareDensity, hardwareDepth, hardwarePinholeDiameter, displayOriginX, displayOriginY,\
displayOriginZ, displayDist, displayRoll, displayYaw, displayPitch, margin, \
retinaNumPixelsWidth, retinaNumPixelsHeight, retinaWidth, retinaHeight, simulatorSampleRate,\
maxTolerance, nearbyPinholes, scheme, samplesPerPixel, samplesPerPinhole, testSuite, testName):
    f = open(configPath, 'w')
    f.write('// INPUT IMAGE\n')
    f.write('imagePath = "{}"\n\n'.format(imagePath))

    f.write('// EYE PARAMETERS\n')
    f.write('eyeFocalDist = {}\n'.format(eyeFocalDist))
    f.write('eyeFocalLength = {}\n'.format(eyeFocalLength))
    f.write('pupilDiameter = {}\n'.format(pupilDiameter))
    f.write('imageDist = {}\n\n'.format(imageDist))

    f.write('// DISPLAY PARAMETERS\n')
    f.write('// A small high res patch\n')
    f.write('displayPreset = "{}"\n'.format(displayPreset))
    f.write('displayPixelWidth = {}\n'.format(displayPixelWidth))
    f.write('displayPixelHeight = {}\n'.format(displayPixelHeight))
    f.write('displayPPI = {}\n\n'.format(displayPPI))

    f.write('// HARDWARE PARAMETERS\n')
    f.write('hardwareType = "{}"\n'.format(hardwareType))
    f.write('hardwareDensity = {}\n'.format(hardwareDensity))
    f.write('hardwareDepth = {}\n'.format(hardwareDepth))
    f.write('hardwarePinholeDiameter = {}\n\n'.format(hardwarePinholeDiameter))

    f.write('// DISPLAY POSE\n')
    f.write('displayOriginX = {}\n'.format(displayOriginX))
    f.write('displayOriginY = {}\n'.format(displayOriginY))
    f.write('displayOriginZ = {}\n'.format(displayOriginZ))
    f.write('displayDist = {}\n\n'.format(displayDist))

    f.write('displayRoll = {}\n'.format(displayRoll))
    f.write('displayYaw = {}\n'.format(displayYaw))
    f.write('displayPitch = {}\n\n'.format(displayPitch))

    f.write('// RETINA PARAMETERS\n')
    f.write('margin = {}\n'.format(margin))
    f.write('retinaNumPixelsWidth = {}\n'.format(retinaNumPixelsWidth))
    f.write('retinaNumPixelsHeight = {}\n'.format(retinaNumPixelsHeight))
    f.write('retinaWidth = {}\n'.format(retinaWidth))
    f.write('retinaHeight = {}\n\n'.format(retinaHeight))

    f.write('// --- Sampling ---\n')
    f.write('// Overall we take <simulatorSampleRate> samples for each retina pixel while simulating and\n')
    f.write('// <nearbyPinholes> * <samplesPerPixel> * <samplesPerPinhole> samples for each pixel during prefiltering\n')
    f.write('// defaults are 64 for simulatorSampleRate, 1 for nearbyPinholes, 1 for samplesPerPixel, 1 for samplesPerPinhole\n')
    f.write('// sampling scheme codes: 0=point to point, 1=point to many, 2=many to many, 3=area to area\n')
    f.write('simulatorSampleRate = {}\n'.format(simulatorSampleRate))
    f.write('maxTolerance = {}\n'.format(maxTolerance))
    f.write('nearbyPinholes = {}\n'.format(nearbyPinholes))
    f.write('scheme = {}\n'.format(scheme))
    f.write('samplesPerPixel = {}\n'.format(samplesPerPixel))
    f.write('samplesPerPinhole = {}\n\n'.format(samplesPerPinhole))

    f.write('// TEST SUITE\n')
    f.write('testSuite = {}\n'.format(testSuite))
    f.write('testName = "{}"'.format(testName))
