import requests
import wget 
import os 

filename = [fn for fn in os.listdir('.') if fn.endswith('.jpg')]
for i in filename:
	try:
		r = requests.get("https://isic-archive.com/api/v1/image?limit=50&offset=0&sort=name&sortdir=1&name=" + i.split('.')[0])
		r = r.json()
		image_id = r[0]['_id']

		r = requests.get("https://isic-archive.com/api/v1/segmentation?limit=50&offset=0&sort=created&sortdir=-1&imageId=" + image_id)
		r = r.json()
		seg_id = r[0]['_id']

		wget.download("https://isic-archive.com/api/v1/segmentation/" + seg_id + "/mask")
	except:
		pass


