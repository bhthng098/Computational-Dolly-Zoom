## Run

You can also run `python main.py -h` to see all arguments.

Required arguments:\
`-s` path/to/image\
`-d` path/to/depth-map\
`-fo` source image's focal length\
`-fmin` the initial focal length to zoom out from
`-fd` desired (maximum) focal length

Optional flags:\
`--show` to display the resulting images using matplotlib\
`--use_fusion_fill` to use the fusion fill image as the source for gap/hole-filling

Results will be saved in `results` folder
### Examples
We have sample images in the `data` folder. Image focal lengths included in name - ie. `data/cube/cube-horiz-13mm.jpeg` has a focal length of 13mm.

To generate a dolly zoom sequence of Rubik's cube images, run:

`python main.py -s data/cube/cube-horiz-13mm.jpeg -d data/cube/cube-horiz-13mm-depth.jpeg -fo 13 -fmin 13 -fd 24 --use_fusion_fill`

## Generate Depth Maps using Tiled ZoeDepth Model
You can use the [Tiled ZoeDepth Colab](https://colab.research.google.com/drive/1wbbXpMC_UUwE3e7Tifq9fYNnd5Rn0zna?usp=sharing#scrollTo=qnfC4dBNbTMh) to generate depth maps. To use, run Code. When the UI pops up, you can click the buttons to install dependencies, upload your own image, process it (longest part), and then save the depth map.

## File Structure
`generateImages.py` generates the desired images using the helper functions contained in `digitalzoom.py` and `dzsynthesis.py`

`digitalzoom.py` contains the digital zoom/reverse digital zoom functions

`dzsynthesis.py` contains both DZ Synthesis functions

`imageocclusion.py` contains image hole filling functions

`utils.py` contains any utility functions.
