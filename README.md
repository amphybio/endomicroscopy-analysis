<!-- https://github.com/amphybio/endomicroscopy-analysis -->

<!-- Add banner here -->
# Motiro

<!-- Add buttons here -->

<!-- ![GitHub release (latest by date including
pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)-->
![GitHub last
commit](https://img.shields.io/github/last-commit/amphybio/endomicroscopy-analysis)
![GitHub
issues](https://img.shields.io/github/issues-raw/amphybio/endomicroscopy-analysis)
![GitHub pull
requests](https://img.shields.io/github/issues-pr/amphybio/endomicroscopy-analysis)
![GitHub](https://img.shields.io/github/license/amphybio/endomicroscopy-analysis)

<!-- Describe your project in brief -->

Motiro - from tupi-guarani, the language of native Brazilians, meaning a reunion
for building - is an unified non-supervised Python-based framework for
pre-processing, segmentation, quantitative and statistical analysis of the
architecture of the colorectal mucosa imaged by probe-based Confocal Laser
Endomicroscopy (pCLE) and, hence, a useful tool for feeding an integrated
database.

<!-- The project title should be self explanotory and try not to make it a
mouthful. (Although exceptions exist-
**awesome-readme-writing-guide-for-open-source-projects** - would have been a
cool name)

Add a cover/banner image for your README. **Why?** Because it easily **grabs
people's attention** and it **looks cool**(*duh!obviously!*).

The best dimensions for the banner is **1280x650px**. You could also use this
for social preview of your repo.

I personally use [**Canva**](https://www.canva.com/) for creating the banner
images. All the basic stuff is **free**(*you won't need the pro version in most
cases*).

There are endless badges that you could use in your projects. And they do depend
on the project. Some of the ones that I commonly use in every projects are given
below.

I use [**Shields IO**](https://shields.io/) for making badges. It is a simple
and easy to use tool that you can use for almost all your badge cravings. -->

<!-- Some badges that you could use -->

<!-- ![GitHub release (latest by date including
pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
: This badge shows the version of the current release.

![GitHub last
commit](https://img.shields.io/github/last-commit/navendu-pottekkat/awesome-readme)
: I think it is self-explanatory. This gives people an idea about how the
  project is being maintained.

![GitHub
issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
: This is a dynamic badge from [**Shields IO**](https://shields.io/) that tracks
  issues in your project and gets updated automatically. It gives the user an
  idea about the issues and they can just click the badge to view the issues.

![GitHub pull
requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)
: This is also a dynamic badge that tracks pull requests. This notifies the
  maintainers of the project when a new pull request comes.

![GitHub All
Releases](https://img.shields.io/github/downloads/navendu-pottekkat/awesome-readme/total):
If you are not like me and your project gets a lot of downloads(*I envy you*)
then you should have a badge that shows the number of downloads! This lets
others know how **Awesome** your project is and is worth contributing to.

![GitHub](https://img.shields.io/github/license/navendu-pottekkat/awesome-readme)
: This shows what kind of open-source license your project uses. This is good
  idea as it lets people know how they can use your project for themselves.

![Tweet](https://img.shields.io/twitter/url?style=flat-square&logo=twitter&url=https%3A%2F%2Fnavendu.me%2Fnsfw-filter%2Findex.html):
This is not essential but it is a cool way to let others know about your
project! Clicking this button automatically opens twitter and writes a tweet
about your project and link to it. All the user has to do is to click tweet.
Isn't that neat? -->

<!-- # Demo-Preview -->

<!-- Add a demo for your project -->

<!-- After you have written about your project, it is a good idea to have a
demo/preview(**video/gif/screenshots** are good options) of your project so that
people can know what to expect in your project. You could also add the demo in
the previous section with the product description.

Here is a random GIF as a placeholder.

![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) -->

# Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table
of contents** or **TOC** as **cool** people say it. This would make it easier
for people to navigate through your README and find exactly what they are
looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this
README. -->

<!-- - [Project Title](#project-title) -->

<!-- - [Demo-Preview](#demo-preview)-->

<!-- - [Table of contents](#table-of-contents) -->
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
   - [Comparative analysis](#comparative-analysis-of-healthy-and-tumor-mucosa)
   - [Statistical characterization](#statistical-characterization-of-the-architecture-of-the-healthy-colorectal-mucosa)
<!-- - [Development](#development) -->
<!-- - [Contribute](#contribute) -->
<!--     <\!-- - [Sponsor](#sponsor) -\-> -->
<!--     - [Adding new features or fixing bugs](#adding-new-features-or-fixing-bugs) -->
- [License](#license)
<!-- - [Footer](#footer) -->

# Dependencies
[(Back to top)](#table-of-contents)

Motiro supports Python 3.6+ and runs on a Linux Operational System. Use the
'requirements.txt' file to install all packages dependencies. The mosaic
function needs [ImageJ](https://imagej.net/Welcome)/[Fiji](https://fiji.sc/)
program with [Register Virtual Stack
Slices](https://imagej.net/Register_Virtual_Stack_Slices) plugin.

# Installation
[(Back to top)](#table-of-contents)

[Download](https://github.com/amphybio/endomicroscopy-analysis/archive/master.zip)
the file and unzip it. Go to the directory where there is the 'requirements.txt'
file and run:
```console
user@term:~$ pip install -r requirements.txt
```
to install the dependencies. After this step, the application will be ready to
use.

<!-- *You might have noticed the **Back to top** button(if not, please notice,
it's right there!). This is a good idea because it makes your README **easy to
navigate.*** The first one should be how to install(how to generally use your
project or set-up for editing in their machine).

This should give the users a concrete idea with instructions on how they can use
your project repo with all the steps.

Following this steps, **they should be able to run this in their device.**

A method I use is after completing the README, I go through the instructions
from scratch and check if it is working. -->

# Usage
[(Back to top)](#table-of-contents)

The folder application has the following structure:


    .
    ├── application                 # Application modules (Python files)
    ├── data-sample                 # Sample files
    │   ├── comparative-sample
    │   │   ├── main
    │   │   │   ├── Sample01        # First patient sample
    │   │   │   .   ├── 0           # pCLE videos that will be used as reference
    │   │   │   .   └── 1           # pCLE videos to be compared
    │   │   │   .
    │   │   │   └── SampleNN        # i-th patient sample
    │   │   │       ├── 0
    │   │   │       └── 1
    │   │   └── sandbox             # Store backup directories to avoid overwriting
    │   └── characterization-sample
    │       ├── main
    │       │   ├── Sample01
    │       │   .   ├── 0
    │       │   .   └── 1
    │       │   .
    │       │   └── SampleNN
    │       └── sandbox
    ├── requirements.txt            # Lists all of a program's dependencies
    ├── LICENSE                     # Code license
    └── README.md


*application* directory contains the Python modules. The module *extract.py*
performs the image analysis in the input files. To see the quick options of
this module run:

```console
user@term:~$ python extract.py -h
```
The command will show the information summarized in the table below:

|    Arguments   |            Description            |
|:--------------:|:---------------------------------:|
| -h, --help     | Show help message and exit        |
| -f, --function | Set a function to call            |
| -p, --path     | Set input file or directory path  |
| -v, --verbose  | Increase log output verbosity     |
| -s, --settings | Settings the input file format    |

The arguments *-f* and *-p* are mandatory to perform the analysis. The functions
available are:

|   Function  |                              Description                              |
|:-----------:|:---------------------------------------------------------------------:|
| comparative | Generate the image frames, pixel intensity and fractal dimension data |
|             |                                                                       |

The module *investigation.py* performs the statistical analysis of the results
generated with the previous module. To see the quick options of this module run:

```console
user@term:~$ python investigation.py -h
```
The command will show the information summarized in the table below:

|    Arguments   |            Description            |
|:--------------:|:---------------------------------:|
| -h, --help     | Show help message and exit        |
| -f, --function | Set a function to call            |
| -p, --path     | Set input file or directory path  |
| -v, --verbose  | Increase log output verbosity     |

The arguments *-f* and *-p* are mandatory to perform the analysis. The functions
available are:

|       Function       |                                                                 Description                                                                |
|:--------------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|
| comparative-analysis | Generate the pixel intensity, fractal dimension and Hellinger distance histograms  and the pixel intensity and fractal dimension Q-Q plots |
|                      |                                                                                                                                            |

The *rvss.py* and *logendo.\** files are used to set the configurations to
stitch images in mosaic and logging, respectively.

In *data-sample* there are some input samples for *compartive analysis* and
*statistical characterization*. Inside of each directory of the classes of
analysis there are the directories '*main*' and '*sandbox*'. The *main*
directory is where the files that will be analysed shall be placed while the
*sandbox* will save directories that can be overwritten during the program are
running.

For the **comparative analysis**, each subdirectory in *main* represent a
patient sample, and the reference healthy videos of a patient must be placed in
the subdirectory *0* while the videos to be compared must be placed in *1*.


## Comparative analysis of healthy and tumor mucosa

### 1. Frames, pixel intensity and fractal dimension data

First step to perform the comparative analysis is generate the frame images from
pCLE videos to produce the pixel intensity and fractal dimension data. The
command below is an example to carry out this steps:

```console
user@term:~$ python extract.py -f comparative -p ../data-sample/comparative-sample/main/
```

Videos with *MP4* and *MPEG* extensions will be listed by default, if different
format is needed the optional argument *-s* can be used to specify the format as follow:

```console
user@term:~$ python extract.py -f comparative -p ../data-sample/comparative-sample/main/ -s .avi .mkv
```

In this example, just videos with *AVI* and *MKV* extensions will be listed. A
list of extensions can be set to search files, is necessary only separate each
extension with a simple space after *-s* argument.

The output of this execution will generate the *frame* folder whereby for each
video input, a folder with frame images named with the video name will be
created. Furthermore, CSV files with the fractal dimension values and other with
pixel intensity of each frame will be produced for each video.

### 2. Generate statistics

After perform the data generation in the previous step, to produce the
statistical analysis and the plots, run the following command:

```console
user@term:~$ python investigation.py -f comparative-analysis -p ../data-sample/comparative-sample/main/
```

This command will use the videos in the reference folder (folder named as *0/*)
for each patient to generate the comparison with the videos in folder *1/*. For
each video will be produce a pixel intensity histogram plot and the Q-Q plot of
pixel intensity as well the distribution histogram and the Q-Q plot of the
fractal dimension will be produced. The Hellinger distance will be calculated
and a histogram plot of the distribution of distances will be generated. For
each video, the related graph starts with the video name followed by the plot
type. (i.e, S01V01-fractal-qq-plot.png)

The plots outside the frame folder use all frames from all videos of the patient
to generate a global statistics of the patient.

## Statistical characterization of the architecture of the healthy colorectal mucosa

### 1. Build the mosaic

The mosaic image can

```console
user@term:~$ python extract.py -f mosaic -p ../data-sample/characterization-sample/main/
```

### 2. Perform crypt morphometry

### 3. Generate statistics

# License
[(Back to top)](#table-of-contents)

<!-- Adding the license to README is a good practice so that people can easily
refer to it.

Make sure you have added a LICENSE file in your project folder. **Shortcut:**
Click add new file in your root of your repo in GitHub > Set file name to
LICENSE > GitHub shows LICENSE templates > Choose the one that best suits your
project!

I personally add the name of the license and provide a link to it like below.
-->

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)

<!-- # Footer -->
<!-- [(Back to top)](#table-of-contents) -->

<!-- <\!-- Let's also add a footer because I love footers and also you **can** use -->
<!-- this to convey important info. -->

<!-- Let's make it an image because by now you have realised that multimedia in -->
<!-- images == cool(*please notice the subtle programming joke). -\-> -->

<!-- Leave a star in GitHub, give a clap in Medium and share this guide if you found -->
<!-- this helpful. -->

<!-- Add the footer here -->

<!--
![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png)
-->
