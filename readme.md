## Notes from Beatrice:
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
