import csv

from ee.geometry import Geometry
from ee.imagecollection import ImageCollection
import geemap
from pathlib import Path
import yaml
import pandas as pd
import os
import shutil
import hashlib
import time
import re
import itertools
import threading
import sys
import ssl
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from platform import python_version
print('Please email feedback to gdex@ucar.edu.\n')

data = [
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201501.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201501.nc','bytes':'47058991516','md5Checksum':'5cab89f598b4785aa994545fec867b58'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201502.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201502.nc','bytes':'42504899360','md5Checksum':'5aee01b43cd9b503da0ea1addcc8ec76'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201503.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201503.nc','bytes':'47058991520','md5Checksum':'68c53ad7c231824a08ef6ecb1a49c2ed'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201504.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201504.nc','bytes':'45540960800','md5Checksum':'a947cff45c2d159661fa1bed93db0ade'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201505.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201505.nc','bytes':'47058991520','md5Checksum':'471f85144da62264d387545cd1533b70'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201506.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201506.nc','bytes':'45540960800','md5Checksum':'311dea4e7d80cb31f82042cc5d00f98b'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201507.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201507.nc','bytes':'47058991520','md5Checksum':'a416b688a327b2b0800e629b4238f86c'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201508.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201508.nc','bytes':'47058991520','md5Checksum':'36ae3c29cc966e65921e07b8942b2530'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201509.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201509.nc','bytes':'45540960800','md5Checksum':'98ace5f4f8eed7006bff247aee7207b0'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201510.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201510.nc','bytes':'47058991520','md5Checksum':'1ee14ed43ddfd1a0d8a0304dc06ea77a'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201511.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201511.nc','bytes':'45540960800','md5Checksum':'599b17726d459d47e38f95f07f3d609a'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201512.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201512.nc','bytes':'47058991520','md5Checksum':'59f8bb4274900f928d2a0ff1f28ad2e9'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201601.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201601.nc','bytes':'47058991516','md5Checksum':'14b9544d8670120bf5e34a0b790cc911'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201602.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201602.nc','bytes':'42504899360','md5Checksum':'926c8384daaf336f83457478cb6748e2'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201603.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201603.nc','bytes':'47058991520','md5Checksum':'bbae1852da0d31ae50fe3cbd0b82390c'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201604.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201604.nc','bytes':'45540960800','md5Checksum':'0bdbdd3cf3ad549c9dd9c81e0aea958b'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201605.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201605.nc','bytes':'47058991520','md5Checksum':'d5dfc9840883aabdb4588b2631acc7f9'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201606.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201606.nc','bytes':'45540960800','md5Checksum':'a3f46ac59234fdde6385a2358dbecd8a'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201607.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201607.nc','bytes':'47058991520','md5Checksum':'33b4fe84a590bdf7e3e1fdc0e3e56786'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201608.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201608.nc','bytes':'47058991520','md5Checksum':'59c6df78e5efe57d84c4bf9b1c575394'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201609.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201609.nc','bytes':'45540960800','md5Checksum':'311f6b5ebb770c5b46df18848c50fa9f'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201610.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201610.nc','bytes':'47058991520','md5Checksum':'d9ca184ea54fab42b23c3b40ee724f3b'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201611.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201611.nc','bytes':'45540960800','md5Checksum':'c1e3a18991ad61a9921a261fc816dbe2'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201612.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201612.nc','bytes':'47058991520','md5Checksum':'e9b4700265a5e8e5f4da8d492a6fb57b'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201701.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201701.nc','bytes':'47058991516','md5Checksum':'e8eec80e2d61d374f79c8c4148d00110'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201702.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201702.nc','bytes':'42504899360','md5Checksum':'aff946c53e48da6e4d4b20122598049b'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201703.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201703.nc','bytes':'47058991520','md5Checksum':'9d7beec600d52ee5fc8ab6fd80e8d63e'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201704.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201704.nc','bytes':'45540960800','md5Checksum':'26c281fb8eed83bff60f260baceb4afc'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201705.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201705.nc','bytes':'47058991520','md5Checksum':'d74a7f4c20d24db290fef37939e6948a'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201706.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201706.nc','bytes':'45540960800','md5Checksum':'df0cc5af3222cdd05b722253f0ef69a2'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201707.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201707.nc','bytes':'47058991520','md5Checksum':'aaf33ecfc28bca3ca68b351952165ebe'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201708.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201708.nc','bytes':'47058991520','md5Checksum':'e647feb81f869a9cb606f744e5c552a9'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201709.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201709.nc','bytes':'45540960800','md5Checksum':'84f864036fcc76e930db57c049da54c8'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201710.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201710.nc','bytes':'47058991520','md5Checksum':'270e79c76f70d09cc130091a1caea031'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201711.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201711.nc','bytes':'45540960800','md5Checksum':'226d90edbb6dc483831dcebab9740c50'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201712.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201712.nc','bytes':'47058991520','md5Checksum':'c1f935aa448913b370dd5b51b40f9b69'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201801.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201801.nc','bytes':'47058991516','md5Checksum':'05e648a4251e59d761ffda3ce354dbc6'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201802.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201802.nc','bytes':'42504899360','md5Checksum':'0bb0eee2b2067c3e4bf1f689d00582de'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201803.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201803.nc','bytes':'47058991520','md5Checksum':'bc887904aee07812309b5f3272d878fd'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201804.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201804.nc','bytes':'45540960800','md5Checksum':'67ee68822c033f6b49132ae274f5582c'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201805.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201805.nc','bytes':'47058991520','md5Checksum':'331fe9d6789a225db426395dc950b16e'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201806.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201806.nc','bytes':'45540960800','md5Checksum':'a52432032f352664818ebe8ada9ae5d7'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201807.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201807.nc','bytes':'47058991520','md5Checksum':'d5a0bfe152cf69db4758066fa122b7c8'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201808.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201808.nc','bytes':'47058991520','md5Checksum':'48c348c3783d13c38d27333018633a62'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201809.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201809.nc','bytes':'45540960800','md5Checksum':'8ea043a36d04c7b2e89e56c1c4438825'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201810.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201810.nc','bytes':'47058991520','md5Checksum':'ba2de542d6be53ea266382c5ddf8fe42'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201811.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201811.nc','bytes':'45540960800','md5Checksum':'f9d9f2d1b9158cdd0b9be130d31f0bb4'},
     {'url':'https://gdex.ucar.edu/api/v1/dataset/382_kumar/version/1.0/file/COMBINE_ACONC_v532_intel_NOAA_fire_201812.nc','filename':'COMBINE_ACONC_v532_intel_NOAA_fire_201812.nc','bytes':'47058991520','md5Checksum':'8d58c0014c6bff6605bc95b5431c04e3'},]



def processArguments():

    args = {}
    args.update({'apiToken': None})
    args.update({'userAgent': 'python/{}/gateway/{}'.format(python_version(), '4.4.18-20250528-234628')})
    args.update({'attemptMax': 10})
    args.update({'initialSleepSeconds': 10})
    args.update({'sleepMultiplier': 3})
    args.update({'sleepMaxSeconds': 900})
    args.update({'insecure': False})

    if '-k' in sys.argv or '--insecure' in sys.argv:
        args.update({'insecure': True})

    if '-h' in sys.argv or '--help' in sys.argv:
        print('Usage: {} [options...]'.format(sys.argv[0]))
        print(' -h, --help        Show usage')
        print(' -k, --insecure    Allow insecure server connections (no certificate check) when using SSL')
        exit(0)

    return args

def executeDownload(download):

    if not os.path.isfile(download.filename):
        attemptAndValidateDownload(download)
        moveDownload(download)
    else:
        download.success = True
        download.valid = True

    reportDownload(download)

def moveDownload(download):

    if download.success and (download.valid or download.vwarning):
        os.rename(download.filenamePart, download.filename)

def reportDownload(download):

    if download.success and download.valid:
        print('{} download successful'.format(download.filename))

    if download.success and not download.valid and download.vwarning:
        print('{} download validation warning: {}'.format(download.filename, download.vwarning))

    if download.success and not download.valid and download.verror:
        print('{} download validation error: {}'.format(download.filename, download.verror))

    if not download.success and download.error:
        print('{} download failed: {}'.format(download.filename, download.error))

def attemptAndValidateDownload(download):

    while download.attempt:
        downloadFile(download)

    if download.success:
        validateFile(download)

def downloadFile(download):

    try :
        startOrResumeDownload(download)
    except HTTPError as error:
        handleHTTPErrorAttempt(download, error)
    except URLError as error:
        handleRecoverableAttempt(download, error)
    except TimeoutError as error:
        handleRecoverableAttempt(download, error)
    except Exception as error:
        handleIrrecoverableAttempt(download, error)
    else:
        handleSuccessfulAttempt(download)

def startOrResumeDownload(download):

    startAnimateDownload('{} downloading:'.format(download.filename))

    if os.path.isfile(download.filenamePart):
        resumeDownloadFile(download)
    else:
        startDownloadFile(download)

def startAnimateDownload(message):
    global animateMessage
    global animateOn

    animateMessage = message
    animateOn = True

    # making the animation run as a daemon thread allows it to
    # exit when the parent (main) is terminated or killed
    t = threading.Thread(daemon=True, target=animateDownload)
    t.start()

def stopAnimateDownload(outcome):
    global animateOutcome
    global animateOn

    animateOutcome = outcome
    animateOn = False

    # wait for animation child process to stop before any parent print
    time.sleep(0.3)

def animateDownload():
    global animateMessage
    global animateOutcome
    global animateOn

    for d in itertools.cycle(['.  ', '.. ', '...', '   ']):

        if not animateOn:
            print('\r{} {}'.format(animateMessage, animateOutcome), flush=True)
            break

        print('\r{} {}'.format(animateMessage, d), end='', flush=True)
        time.sleep(0.2)

def resumeDownloadFile(download):

    request = createRequest(download, createResumeHeaders(download))
    readFile(download, request)

def startDownloadFile(download):

    request = createRequest(download, createStartHeaders(download))
    readFile(download, request)

def createResumeHeaders(download):

    headers = createStartHeaders(download)
    headers.update(createRangeHeader(download))

    return headers

def createRequest(download, headers):

    request = urllib.request.Request(download.url, headers=headers)

    return request

def createStartHeaders(download):

    headers = {}
    headers.update(createUserAgentHeader(download))

    if download.apiToken:
        headers.update(createAuthorizationHeader(download))

    return headers

def createUserAgentHeader(download):

    return {'User-agent': download.userAgent}

def createAuthorizationHeader(download):

    return {'Authorization': 'api-token {}'.format(download.apiToken)}

def createRangeHeader(download):

    start = os.path.getsize(download.filenamePart)
    header = {'Range': 'bytes={}-'.format(start)}

    return header

def readFile(download, request):

    context = createSSLContext(download)

    with urllib.request.urlopen(request, context=context) as response, open(download.filenamePart, 'ab') as fh:
        collectResponseHeaders(download, response)
        shutil.copyfileobj(response, fh)

def createSSLContext(download):

    # See:
    #      https://docs.python.org/3/library/urllib.request.html
    #      https://docs.python.org/3/library/http.client.html#http.client.HTTPSConnection
    #      https://docs.python.org/3/library/ssl.html#ssl.SSLContext
    #
    # Excerpts:

    #      If context is specified it must be a ssl.SSLContext instance...
    #      http.client.HTTPSConnection performs all the necessary certificate and hostname checks by default.

    if download.insecure:
        return ssl._create_unverified_context()

    return None

def collectResponseHeaders(download, response):

    download.responseHeaders = response.info()
    if download.responseHeaders.get('ETag'):
        download.etag = download.responseHeaders.get('ETag').strip('"')

def handleHTTPErrorAttempt(download, httpError):

    if httpError.code == 416: # 416 is Range Not Satisfiable
        # likely the file completely downloaded and validation was interrupted,
        # therefore calling it successfully downloaded and allowing validation
        # to say otherwise
        handleSuccessfulAttempt(download)
    else:
        handleRecoverableAttempt(download, httpError)

def handleRecoverableAttempt(download, error):

    stopAnimateDownload('error')

    print('failure on attempt {} downloading {}: {}'.format(download.attemptNumber, download.filename, error))

    if download.attemptNumber < download.attemptMax:
        sleepBeforeNextAttempt(download)
        download.attemptNumber += 1
    else:
        download.attempt = False
        download.error = error

def sleepBeforeNextAttempt(download):

    sleepSeconds = download.initialSleepSeconds * (download.sleepMultiplier ** (download.attemptNumber - 1))

    if sleepSeconds > download.sleepMaxSeconds:
        sleepSeconds = download.sleepMaxSeconds

    print('waiting {} seconds before next attempt to download {}'.format(sleepSeconds, download.filename))
    time.sleep(sleepSeconds)

def handleIrrecoverableAttempt(download, error):

    stopAnimateDownload('error')

    download.attempt = False
    download.error = error

def handleSuccessfulAttempt(download):

    stopAnimateDownload('done')

    download.attempt = False
    download.success = True

def validateFile(download):

    try:
        validateAllSteps(download)
    except InvalidDownload as error:
        download.valid = False
        download.vwarning = str(error)
    except Exception as error:
        download.valid = False
        download.verror = error
    else:
        download.valid = True

def validateAllSteps(download):

    verrorData = validatePerData(download)
    verrorEtag = validatePerEtag(download)
    verrorStale = validateStaleness(download)

    if verrorData and verrorEtag:
        raise verrorData

    if verrorStale:
        raise verrorStale

def validatePerData(download):

    try:
        validateBytes(download)
        validateChecksum(download)
    except InvalidDownload as error:
        return error
    else:
        return None

def validateBytes(download):

    size = os.path.getsize(download.filenamePart)
    if not download.bytes == size:
        raise InvalidSizeValue(download, size)

def validateChecksum(download):

    if download.md5Checksum:
        md5Checksum = readMd5Checksum(download)
        if not download.md5Checksum == md5Checksum:
            raise InvalidChecksumValue(download, md5Checksum)
    else:
        raise UnableToPerformChecksum(download)

def readMd5Checksum(download):

    hash_md5 = hashlib.md5()

    with open(download.filenamePart, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()

def validatePerEtag(download):

    try:
        validateChecksumEtag(download)
    except InvalidDownload as error:
        return error
    else:
        return None

def validateChecksumEtag(download):

    if isEtagChecksum(download):
        md5Checksum = readMd5Checksum(download)
        if not download.etag == md5Checksum:
            raise InvalidChecksumValuePerEtag(download, md5Checksum)
    else:
        raise UnableToPerformChecksum(download)

def isEtagChecksum(download):

    return download.etag and re.fullmatch(r'[a-z0-9]+', download.etag)

def validateStaleness(download):

    try:
        validateStaleChecksum(download)
    except InvalidDownload as error:
        return error
    else:
        return None

def validateStaleChecksum(download):

    if isEtagChecksum(download):
        if not download.md5Checksum or download.md5Checksum != download.etag:
            raise StaleChecksumValue(download)

class InvalidDownload(Exception):

    pass

class InvalidSizeValue(InvalidDownload):

    def __init__(self, download, actual):
        super().__init__('invalid byte size: downloaded file is {} bytes but should be {}'.format(actual, download.bytes))

class InvalidChecksumValue(InvalidDownload):

    def __init__(self, download, actual):
        super().__init__('invalid checksum: downloaded file is {} but should be {}'.format(actual, download.md5Checksum))

class InvalidChecksumValuePerEtag(InvalidDownload):

    def __init__(self, download, actual):
        super().__init__('invalid checksum: downloaded file is {} but should be {} according to server'.format(actual, download.etag))

class UnableToPerformChecksum(InvalidDownload):

    def __init__(self, download):
        super().__init__('cannot verify checksum')

class StaleChecksumValue(InvalidDownload):

    def __init__(self, download):
        super().__init__('checksum value has changed')

class Download():

    def __init__(self, args, datum):

        self.apiToken = args.get('apiToken')
        self.userAgent = args.get('userAgent')
        self.attemptMax = args.get('attemptMax')
        self.initialSleepSeconds = args.get('initialSleepSeconds')
        self.sleepMultiplier = args.get('sleepMultiplier')
        self.sleepMaxSeconds = args.get('sleepMaxSeconds')
        self.insecure = args.get('insecure')

        self.url = datum.get('url')
        self.filename = datum.get('filename')
        self.bytes = int(datum.get('bytes'))
        self.md5Checksum = datum.get('md5Checksum')

        self.filenamePart = self.filename + '.part'
        self.success = False
        self.attempt = True
        self.attemptNumber = 1
        self.responseHeaders = {}
        self.etag = None
        self.error = None
        self.valid = False
        self.vwarning = None
        self.verror = None

    def __str__(self):
        return f'url: {self.url}, filename: {self.filename}, bytes: {self.bytes}, md5Checksum: {self.md5Checksum}'

def fmt_sensor_data(sensors):
    sensors["Date GMT"] = pd.to_datetime(sensors["Date GMT"])
    sensors["Time GMT"] = pd.to_datetime(sensors["Time GMT"], format="%H:%M").dt.hour
    sensors["time"] = sensors["Date GMT"] + pd.to_timedelta(sensors["Time GMT"], unit="h")
    sensors.rename(columns={"Latitude": "latitude"}, inplace=True)
    sensors.rename(columns={"Longitude": "longitude"}, inplace=True)
    sensors.rename(columns={"Sample Measurement": "pm25"}, inplace=True)
    sensors["id"] = (
        sensors["State Code"].astype(str) +
        sensors["County Code"].astype(str) +
        sensors["Site Num"].astype(str)
    )

    sensors = sensors[["id", "latitude", "longitude", "time", "pm25"]]

    return sensors

def get_point_data(s, fcams):
    unique_sensors = s[["longitude", "latitude"]].drop_duplicates()

    all_point_data = []

    for _, row in unique_sensors.iterrows():
        point = Geometry.Point([row['longitude'], row['latitude']])
        raw = fcams.getRegion(point, 10_000).getInfo()
        headers = raw[0]
        data = raw[1:]
        df = pd.DataFrame(data, columns=headers)
        df["latitude"] = row["latitude"]
        df["longitude"] = row["longitude"]
        all_point_data.append(df)

    point_data = pd.concat(all_point_data, ignore_index=True)
    point_data["time"] = pd.to_datetime(point_data["time"], unit="ms")
    point_data.rename(columns={"particulate_matter_d_less_than_25_um_surface": "pm25"}, inplace=True)
    point_data["pm25"] *= 1_000_000_000
   # point_data = point_data[point_data["id"].str.contains("F000", na=False)]

    return point_data



def process_fire(fire_name, fire_data, sensors, cams, writer):
    name = fire_name
    latitude = fire_data.get("latitude")
    longitude = fire_data.get("longitude")
    start = pd.to_datetime(fire_data.get("start"))
    end = pd.to_datetime(fire_data.get("end"))

    print(f"Processing {name}")
    count = 0

    s = sensors[
        (abs(sensors["latitude"] - latitude) < 0.5) &
        (abs(sensors["longitude"] - longitude) < 0.5) &
        (sensors["time"] >= start) &
        (sensors["time"] <= end)
    ]

    if s.empty:
        print("No sensor data available")
        return

    region = Geometry.Rectangle([
        longitude - 0.5,
        latitude - 0.5,
        longitude + 0.5,
        latitude + 0.5
    ])

    fcams = (
        cams
        .filterBounds(region)
        .filterDate(start, end)
    )

    point_data = get_point_data(s, fcams)

    merged = pd.merge(
        s,
        point_data,
        on=["latitude", "longitude", "time"],
        how="inner",
        suffixes=("_sensor", "_sat")
    )

    for _, row in merged.iterrows():
        writer.writerow([
            name,
            row["id_sensor"],
            row["id_sat"],
            row["latitude"],
            row["longitude"],
            row["time"],
            row["pm25_sensor"],
            row["pm25_sat"],
        ])
        count += 1

    print(f"Done processing {name}, {count} matches")

import xarray as xr
from pyproj import Proj, Transformer
import datetime

def main() -> None:
    ncdf_path = Path("data/netcdf/nc.nc")
    
    ds = xr.open_dataset(ncdf_path, decode_cf=False)  

    lat, lon = 46.8721, -113.9940

    # Set up Lambert Conformal projection based on dataset attributes
    proj = Proj(proj='lcc',
                lat_1=ds.attrs['P_ALP'],
                lat_2=ds.attrs['P_BET'],
                lat_0=ds.attrs['YCENT'],
                lon_0=ds.attrs['XCENT'],
                x_0=0, y_0=0, ellps='sphere')

    transformer = Transformer.from_proj("epsg:4326", proj, always_xy=True)

    # Convert lon/lat to x/y (in meters)
    x, y = transformer.transform(lon, lat)

    # Convert x/y to COL/ROW
    XORIG = ds.attrs['XORIG']
    YORIG = ds.attrs['YORIG']
    XCELL = ds.attrs['XCELL']
    YCELL = ds.attrs['YCELL']

    col = int((x - XORIG) / XCELL)
    row = int((y - YORIG) / YCELL)

    # Choose a timestep (e.g., 0 = start of simulation)
    t = 0

    # Extract PM2.5 value
    pm25 = ds['PM25_TOT'].isel(TSTEP=t, LAY=0, ROW=row, COL=col).values.item()

    # Optionally get the timestamp
    base_date = datetime.datetime.strptime(str(ds.attrs['SDATE']), "%Y%j")
    step_hours = int(str(ds.attrs['TSTEP'])[:2])  # assuming TSTEP = 10000 → 1hr
    timestamp = base_date + datetime.timedelta(hours=step_hours * t)

    print(f"PM2.5 at Missoula (lat={lat}, lon={lon}) on {timestamp} = {pm25} µg/m³")

        # geemap.ee_initialize()

    # args = processArguments()
    #
    # for d in data:
    #     executeDownload(Download(args, d))


    #
    # fire_dir = Path("data/fires")
    # sensor_dir = Path("data/sensor")
    # out_path = Path("fout.csv")
    #
    # cams = (
    #     ImageCollection("ECMWF/CAMS/NRT")
    #     .select("particulate_matter_d_less_than_25_um_surface")
    # )
    #
    # with out_path.open("w", newline="", buffering=100) as f_out:
    #     writer = csv.writer(f_out)
    #     writer.writerow(
    #         [
    #             "fire_name",
    #             "sensor_id",
    #             "sat_id",
    #             "lat",
    #             "lon",
    #             "time",
    #             "sensor_pm25",
    #             "sat_pm25",
    #         ]
    #     )
    #
    #     for fire_path in fire_dir.glob("us_fire_*_1e7.yml"):
    #         year = fire_path.stem.split("_")[2]
    #         for t in ["88101", "88502"]:
    #             file = f"{sensor_dir}/hourly_{t}_{year}.csv"
    #             print(f"Reading {file}")
    #             sensors = pd.read_csv(file, header=0, skiprows=0)
    #             sensors = fmt_sensor_data(sensors)
    #             print(f"Processing {file}")
    #             with fire_path.open() as f:
    #                 fire_file = yaml.safe_load(f)
    #                 fires = {k: v for k, v in fire_file.items() if k not in ("output_bucket", "rectangular_size", "year")}
    #                 for fire_name, fire_data in fires.items():
    #                     try:
    #                         process_fire(fire_name, fire_data, sensors, cams, writer)
    #                     except Exception as e:
    #                         print(e)
    #

if __name__ == "__main__":
    main()



