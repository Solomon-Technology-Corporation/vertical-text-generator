# vertical-text-generator
vertical text image generator from recognition dataset, bg_img is sampled from SynthText's background image list.

# Requirement

```
# if you use Python3 on Ubuntu, please use "pip3" instead.
pip install trdg
pip install shapely
pip install opencv-python
```

# How to use

1. Generate Fake images containing only text from [TRDG](https://github.com/Belval/TextRecognitionDataGenerator), then it will create a folder "results" containing N images inside.
```
# to create N=1000 images,
trdg -l en -rs -let -num -sym -t 8 -e jpg -k 2 -rk -b 3 -na 2 -obb 1 -f 64 -or 0 -fi -id images -c N #change N to your desired number of samples, or see trdg --help for more details
```
2. Run vertical text generator with the generated fake images.
```
Ubuntu: python vertical_textdet_generator.py -text "../out/" -bg "../bg_img/" -opath "vtg_out/" -n 10 -lp 2
```
