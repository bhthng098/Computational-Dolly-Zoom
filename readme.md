# Notes from Beatrice:

## Re-organized Files
I reorganized our functions into 4 differente files:

`generateImages.py` generates the desired images using the helper functions contained in `digitalzoom.py` and `dzsynthesis.py`\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**NOTE: `main.py` should only ever call functions in this file** and not the below helper files:

`digitalzoom.py` contains the digital zoom/reverse digital zoom functions

`dzsynthesis.py` contains both DZ Synthesis functions

`utils.py` contains any utility functions. Right now it just has `show_2_images_side_by_side`

## Argparser
I added an arg parser for terminal commands!

Run `python main.py -h` to see the arguments

Optional arguments:\
`--save` to save the images\
`--quiet` to not display the images using matplotlib

### Examples:
To generate all Arkansas images from scratch and to save them, run:

`python main.py -s ./data/arkansas/arkansas-50mm.jpeg -d ./data/arkansas/arkansas-50mm_depth.jpeg --save`

To generate and save all Arkansas images using previously-generated digital zoom images:

`python main.py -s ./data/arkansas/arkansas-50mm.jpeg -d ./data/arkansas/arkansas-50mm_depth.jpeg --izoom ./results/arkansas/50-85/arkansas-50mm-zoome-50-85.jpeg --dzoom ./results/arkansas/50-85/arkansas-50mm_depth-zoomed-50-85.jpeg --save`

## Generate Depth Maps using Tiled ZoeDepth Model
You can use the [Tiled ZoeDepth Colab](https://colab.research.google.com/drive/1wbbXpMC_UUwE3e7Tifq9fYNnd5Rn0zna?usp=sharing#scrollTo=qnfC4dBNbTMh) to generate depth maps. To use, run Code. When the UI pops up, you can click the buttons to install dependencies, upload your own image, process it (longest part), and then save the depth map.