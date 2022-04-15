# Python-Macduff
This is a Python port of [Macduff](https://github.com/ryanfb/macduff), a tool
for finding the Macbeth ColorChecker chart in an image.

The translation to python was done by Andrew Port and was supported by
[Rare](https://rare.org) as part of work done for the
[Fish Forever](http://www.fishforever.org/) project.

## Changes from the original algorithm:
    * An (in-code) parameter `MIN_RELATIVE_SQUARE_SIZE` has been added as a
    work-around for an issue where the algorithm would choke on images where
    the ColorChecker was smaller than a certain hard-coded size relative to
    the image dimensions.

    * An optional `patch_size` parameter has been added to give better results
    when the approximate (within rtol=25%) pixel-width of the color patches is
    known.

    * Several additional colorchecker color value options are now included.  The
    default has been changed to those values provided by xrite for the "passport"
    colorchecker.

## Usage
  
    # if pixel-width of color patches is unknown,  
    $ python macduff.py examples/test.jpg result.png > result.csv  

    # if pixel-width of color patches is known to be, e.g. 65,  
    $ python macduff.py examples/test.jpg result.png 65 > result.csv  

## DESCRIPTION

![Macduff result](https://ryanfb.s3.amazonaws.com/images/macduff.png)

Macduff will try its best to find your ColorChecker. If you specify an output
image, it will be written with the "found" ColorChecker overlaid on the input
image with circles on each patch (the outer circle is the "reference" value,
the inner circle is the average value from the actual image). Macduff outputs
various useless debug info on stderr, and useful information in CSV-style
on stdout. The first 24 lines will be the ColorChecker patch locations and
average values:

    x,y,r,g,b

The last two lines contain the patch square size (i.e. you can feed another
program this and the location and safely use a square of `size` with the top
left corner at `x-size/2,y-size/2` for each patch) and error against the
reference chart. The patches are output in row order from the typical
ColorChecker orientation ("dark skin" top left, "black" bottom right):

![ColorChecker layout](https://ryanfb.s3.amazonaws.com/images/CC_Avg20_orig_layout.png)

See also: [Automatic ColorChecker Detection, a Survey](http://ryanfb.github.io/etc/2015/07/08/automatic_colorchecker_detection.html)

## LICENSE
The original Macduff code is protected under the 3-clause BSD and includes some code taken from OpenCV.  See LICENSE.TXT.
