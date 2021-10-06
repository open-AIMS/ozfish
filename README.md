[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 3.0 Australian License][cc-by].

[![CC BY 3.0 AU][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/3.0/
[cc-by-image]: https://i.creativecommons.org/l/by/3.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%203.0-lightgrey.svg

## OzFish dataset

OzFish is a collection of ~80k fish crops, ~45k bounding box annotations derived from Baited Remote Underwater Video Stations (BRUVS) and comprised of 70 families, 200 genera and 507 species of fish. This dataset is completely open and free to use for advancing machine learning for the classification of fish from underwater imagery. 

To cite this dataset use the following: 

Australian Institute of Marine Science (AIMS), University of Western Australia (UWA) and Curtin University. (2019), OzFish Dataset - Machine learning dataset for Baited Remote Underwater Video Stations, [https://doi.org/10.25845/5e28f062c5097](https://doi.org/10.25845/5e28f062c5097)

For more information see [metadata](https://doi.org/10.25845/5e28f062c5097).

### Raw Video 

[Raw Stero BRUVS](https://data.pawsey.org.au/public/?path=/FDFML/videos)

### Crops

![Fish Crops](https://open-AIMS.github.io/ozfish/fishcrops.png?raw=true "Fish Crops")

Fish crops are from frames in videos where we had associated measurements with fish in the frames. Using the measurements we were able estimate a bounding box for the fish, the images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/crops) and metadata [here](https://data.pawsey.org.au/download/FDFML/metadata/crop_metadata.csv).

The fish crops have an associated metadata file which links the species, genus, family annotation to the crop.

```markdown
uid,file_name,family,genus,species
1,A000001_L.avi.5107.806.371.922.448.png,Scaridae,Chlorurus,capistratoides
```

### Frames

Frames are extracted from the videos where we have an event measure measurement, and an associated fish label. The images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/frames) and metadata [here](https://data.pawsey.org.au/download/FDFML/metadata/frame_metadata.csv).

Frames have an associated metadata file which links the species, genus, family annotation fish in the frame.

x0, x1 = pixels from left of image

y0, y1 = pixels from top of image

```markdown
uid,file_name,x0,y0,x1,y1,family,genus,species
1,A000001_L.avi.5107.png,806,371,922,448,Scaridae,Chlorurus,capistratoides
```

### Bounding Box Annotations

![Bounding box annotations](https://open-AIMS.github.io/ozfish/bounding-box-annotations.png?raw=true "Bounding box annotations")

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

### Bounding Box Species Annotations

Bounding box annotations were generated on the Sagemaker Ground Truth Platform, using multiple observers and combining the results. Bounding boxes were then labelled to species level (using vgg annotator) by an ecologist for fish that were identifiable in the frame. Fish which were not identifiable to species were left with the label "fish". The images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/frames) and metadata [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/speciesboxes).

### Bounding Box Tail Annotations

![Bounding box tail annotations](https://aims.github.io/ozfish/fish_tails.png?raw=true "Bounding box annotations")

Bounding boxes were applied to the tails of fish (using vgg annotator) that were identifiable under a set of seven categories: 'Emarginate', 'Ornate lunated', 'Truncated', 'Lunate', 'Heterocercal', 'Forked', 'Rounded'. Tails which could not be identified were not bounding boxed or labelled. The images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/frames) and metadata [here](https://data.pawsey.org.au/download/FDFML/labelled/fishtails/FishTails_via.json).

### Fish measurement files

These files are exports from the event measure software which give pixel locations for nose and tail of fish which were measured, and a measurement in mm for the given fish. The images are available [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/frames) and metadata [here](https://data.pawsey.org.au/public/?path=/FDFML/labelled/measurementfiles). 

The following is an example python snippet for reading a measurement file and drawing a measurement line.

```markdown
import cv2  
import pandas
import os

df = pandas.read_csv("../A_lengths.csv")
frames_path = "../OzFishFrames/"

point_pairs = df.ImagePtPair.unique()
line_thickness = 2

for point in point_pairs:

    frame = df[(df.ImagePtPair == point)]["OzFishFrame"].values[0]
    image_path = os.path.join(frames_path, frame)
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

    lx0 = int(df[(df.ImagePtPair == point) & (df.Index == 0)]["Lx"].values[0])
    ly0 = int(df[(df.ImagePtPair == point) & (df.Index == 0)]["Ly"].values[0])
    lx1 = int(df[(df.ImagePtPair == point) & (df.Index == 1)]["Lx"].values[0])
    ly1 = int(df[(df.ImagePtPair == point) & (df.Index == 1)]["Ly"].values[0])
    print(lx0, ly0, lx1, ly1, image_path)
    cv2.line(image, (lx0, ly0), (lx1, ly1), (0, 255, 0), thickness=line_thickness)

    cv2.imshow('Measurement in mm', image)

    if cv2.waitKey() == ord('q'):
        exit(0)
```
