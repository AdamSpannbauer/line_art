# Line Art<sup>1</sup>

Draw an input image using a random walk.  Comes in 2 flavors: (1) Black/White & (2) Color.  Presented with little description, documentation, usage information.

The color version uses the dominant color finding process shown [here](https://adamspannbauer.github.io/2018/03/02/app-icon-dominant-colors/) (with some modifications to avoid 'unwanted' colors).

Built with:

* Python (for uhh... everything)
* OpenCV (for segmenting/drawing<sup>2</sup>)
* Sci-kit Learn (for finding dominant colors)

## Example Input/Output

### Color

Output:

<p align='center'><img src='readme/py_panda.gif' width='60%'></p>



Input:

<p align='center'><img src='images/py_pandas_2.png' width='60%'></p>



### Black/White

Output:

<p align='center'><img src='readme/hourglass.gif' width='40%'></p>



Input:

<p align='center'><img src='images/hourglass_mask.jpg' width='40%'></p>

<sub><sup>1</sup>Not sure if technically lines or technically art</sub>

<sub><sup>2</sup>I should prolly learn p5.js instead of using OpenCV for these drawing type projects</sub>