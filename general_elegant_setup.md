# Installing Elegant: tools and pipelines for C. elegans data
Author: Nicolette Laird

Date: 2019-07-16

http://zplab.wustl.edu

## 1. Install Miniconda
- Download [Miniconda for Python 3](https://docs.conda.io/en/latest/miniconda.html). Make sure you download the Python 3.7 installer that best fits your machine.

- Install miniconda:
  In your terminal window, run:
  ```
  bash Miniconda3-latest-*-x86_64.sh -b -p $HOME/miniconda
  rehash
  rm Miniconda3-latest-*-x86_64.sh
  ```
- Follow the prompts on the installer screens
- Close and re-open your terminal window to enable changes

If you need help, look through the [Miniconda Documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Here is a quick guide to [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), although for basic use you don't need
to know any of this.

## 2. Setup Python Environment
- Create a new environment with specific packages:
   
  Install the basic suite of zplab tools into the root miniconda environment.  
  
  In a terminal window, run:
  
  -**NOTE:** replace myenv to the environment name you want
  ```
  cat > user_env.yml << EOF
  channels:
      - defaults
      - conda-forge

  dependencies:
      - ipython
      - numpy
      - scipy
      - scikit-learn
      - pip:
          - scikit-image
          - git+https://github.com/zplab/freeimage-py
          - git+https://github.com/zplab/zplib
          - git+https://github.com/zplab/RisWidget
          - git+https://github.com/zplab/elegant
  EOF
  pip install --upgrade pip
  conda update conda
  
  conda env update -n myenv -f user_env.yml
  pip install celiagg --global-option=--no-text-rendering
  
  rm user_env.yml
  ```
- Activate your new environment!
  ```
  conda activate myenv
  ```

**NOTE:** if you ever need to update to the latest version of zplab-specific libraries, run lines like the following:
  ```
  pip install --upgrade git+https://github.com/zplab/freeimage-py
  ```

## 3. Start annotating!
- Use Elegant to annotate centerlines and straighten images of worms!

  To begin the general pose annotator, open a terminal and run
  ```
  general_pose_annotator
  ```
  A RisWidget window should appear. You can drag and drop images into the window to begin annotating.
  If you need a specific pixels per micron or load up specific images via command line use:
  ```
    general_pose_annotator `bc -l <<< 'pixels_per_micron'` 'path to image 1' 'path to image 2'
  ```
  **NOTE:** you can add as many images as you want.
  
 You can also access the annotator via ipython.
 See [Elegant Documentation](https://github.com/zplab/elegant) for more help.
  
 
