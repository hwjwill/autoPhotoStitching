# Auto-Photo Stitching

This project aimes to explore applications of homography transformation. This project is divided into four parts. Part 1 rectifies an image into desired orientation or shape. It can be used to better visualize image or patterns that is shot from side. Part 2 allows manual photo stitching. User will need to manually select conrespondences of kpoints between two images and save them in csv file for the program to read and stitch. Part 3 did auto-stitching. It detects keypoints using Harris corner detection and other algorithms listed in [this paper](http://inst.eecs.berkeley.edu/~cs194-26/fa16/Papers/MOPS.pdf) by Brown et al to automatically stitch two photos. Part 4 automatically detects similar images that can be stithced together from an array of images, based on number of matching keypoints, and follows procedure in part 3 to stitch them together. For more information, please visit [here](https://inst.eecs.berkeley.edu/~cs194-26/fa16/upload/files/proj7B/cs194-26-acm/).

![Example](https://inst.eecs.berkeley.edu/~cs194-26/fa16/upload/files/proj7A/cs194-26-acm/mStitch.jpg)

# To run code

Please see comments in main.m for further definition of parameters
```
main()
``` 