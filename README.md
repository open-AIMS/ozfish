[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 3.0 Australian License][cc-by].

[![CC BY 3.0 AU][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/3.0/
[cc-by-image]: https://i.creativecommons.org/l/by/3.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%203.0-lightgrey.svg

## OzFish dataset

OzFish is a collection of ~80k fish crops, ~45k bounding box annotations derived from Baited Remote Underwater Video Stations (BRUVS). This dataset is completely open and free to use for advancing machine learning for the classification of fish from underwater imagery. 

To cite this dataset use the following: 

Australian Institute of Marine Science (AIMS), University of Western Australia (UWA) and Curtin University. (2019), OzFish Dataset - Machine learning dataset for Baited Remote Underwater Video Stations , https://doi.org/10.25845/5e28f062c5097

### Crops

![Fish Crops](/fishcrops.png?raw=true "Fish Crops")

Fish crops are from frames in videos where we had associated measurements with fish in the frames. Using the measurements we were able estimate a bounding box for the fish, the images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/crops) and metadata [here](https://data.pawsey.org.au/download/FDFML/metadata/crop_metadata.csv).

The fish crops have an associated metadata file which links the species, genus, family annotation to the crop.

```markdown
uid,file_name,family,genus,species
1,A000001_L.avi.5107.806.371.922.448.png,Scaridae,Chlorurus,capistratoides
```

### Frames

Frames are extracted from the videos where we have an event measure measurement, and an assocaited fish label. The images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/frames) and metadata [here](https://data.pawsey.org.au/download/FDFML/metadata/frame_metadata.csv).

Frames have an associated metadata file which links the species, genus, family annotation fish in the frame.

x0, x1 = pixels from left of image
y0, y1 = pixels from top of image

```markdown
uid,file_name,x0,y0,x1,y1,family,genus,species
1,A000001_L.avi.5107.png,806,371,922,448,Scaridae,Chlorurus,capistratoides
```

### Bounding Box Annotations

![Bounding box annotations](/bounding-box-annotations.png?raw=true "Bounding box annotations")

Bounding box annotations were generated on the Sagemaker Ground Truth Platform, using multiple observers and combining the results. Unlike the crops, frames and videos, these annotations are fish/no-fish only and have no species/genus/family labels. The images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/frames) and metadata [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/manifests).

Bounding boxes have associated JSON metadata.

```markdown
{
    "source-ref":"E000501_R.MP4.31568.png",
    "20191014":{
        "annotations":[
            {"class_id":0,"width":139,"top":306,"height":84,"left":588.5},{"class_id":0,"width":229.5,"top":357,"height":331,"left":1151},{"class_id":0,"width":198.5,"top":745.5,"height":271,"left":823},{"class_id":0,"width":159.5,"top":806,"height":148.5,"left":0},{"class_id":0,"width":1014,"top":399.5,"height":395,"left":108.5}
            ],
            "image_size":[{"width":1920,"depth":3,"height":1080}]},
            "20191014-metadata":{
                "class-map":{"0":"fish"},
                "human-annotated":"yes",
                "objects":[{"confidence":0.27},{"confidence":0.27},{"confidence":0.2},{"confidence":0.27},{"confidence":0.28}],
                "creation-date":"2019-10-15T05:40:28.278830",
                "type":"groundtruth/object-detection"
            }
    }
```

The following is an example python snippet for reading the json.

```markdown
manifest = "output.manifest"

with open(manifest) as json_file:
    for line in json_file:
        j_content = json.loads(line)
        image_name = os.path.basename(j_content["source-ref"])
        
        print(image_name)
        
        annotations = j_content["20191028"]["annotations"]

        for annotation in annotations:
            print(annotation["left"], annotation["top"], annotation["width"], annotation["height"])
```
