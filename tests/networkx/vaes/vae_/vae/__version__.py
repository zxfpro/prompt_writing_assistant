from warnings import catch_warnings
with catch_warnings(record=True):
    import json

version_json = '''
{
 "version": "0.0.2"
}
'''

def get_versions():
    return json.loads(version_json)
