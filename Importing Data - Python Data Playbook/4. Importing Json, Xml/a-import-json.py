import json

# Importing JSON with load
with open('posts-100.json', 'r') as f:
    posts_json = json.load(f)
type(posts_json)
len(posts_json)
len(posts_json[0])
print(posts_json[0]['Id'], posts_json[0]['Title'])

# Importing JSON with loads
json_loaded = json.loads(json_original)
json_loaded
type(json_loaded)
print(json.dumps(json_loaded))
print(json.dumps(json_loaded, indent=2))
