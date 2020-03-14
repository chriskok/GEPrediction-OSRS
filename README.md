# OSRS Grand Exchange Price Prediction

Start-to-end project where we attempt to harness the power of machine learning to predict Old-school Runescape Grand Exchange prices.

## Getting started
The journey so far has been documented in a series of Youtube videos found here: 
1. [Part 1](https://youtu.be/D5TmBcpgm7k) - Setup and initial trial
2. [Part 2](https://youtu.be/U453OSC8dkc) - Data Collection
3. [Part 3](https://www.youtube.com/watch?v=QUvyroIH3jI&list=PLX9loFun2zNmri7jHhLs7NV76wcGRzI45&index=3) - Feature Engineering and Selection
4. [Part 4](https://www.youtube.com/watch?v=LzVN2MGqz2w&list=PLX9loFun2zNmri7jHhLs7NV76wcGRzI45&index=4) - Hyperparameter Tuning, Application and API

### Prerequisites

- [Python](https://www.python.org/downloads/)
- [Tensorflow](https://www.tensorflow.org/install)
- [Jupyter Notebook](https://jupyter.org/install)
- Python dependencies:
  - requests
  - json
  - csv
  - time
  - matplotlib 
  - numpy
  - pandas
  - flask
  
### Creating you own models
1. Change the items_to_predict array in the main() function to the items you wish to use.
2. Then, run:
```
python models.py
```
3. You should see the .h5 model file created in the models folder along with features.txt file in the models/features folder

### Applying the created models
1. Make sure you have the latest data stored in data/rsbuddy or change the path of DATA_FOLDER in line 101 of application.py
2. Change the items_to_predict array in the main() function to match the models you created/have.
3. Then, run:
```
python application.py
```
4. You should see a .csv file created (or have data appended to) in the name of that item in data/predictions.

### Running the flask app
1. Change items in items_predicted array in index() to match the items that you've predicted on
2. Run:
```
python flask-app.py
```
3. Go to localhost:80 and see your results!

### Running the jupyter notebooks

1. Move the preferred notebook out of the Notebooks foler to the main directory 
2. Run the following command: 
```
jupyter notebook
```

### Scraping your own data

If you wish to scrape your own data the way I've been doing it, run the following script every 2 minutes (for osbuddy):
```
python osbuddy-ge-scraper.py
```
OR every 30 minutes (for rsbuddy):

```
python rsbuddy-ge-scraper.py
```
You can do this automatically by using [crontab](http://man7.org/linux/man-pages/man5/crontab.5.html) if you're on a Linux machine or [windows scheduler](https://www.windowscentral.com/how-create-automated-task-using-task-scheduler-windows-10) if you're on a Windows machine. 

### Contributions

* Please email me at billnyetheai@gmail.com or message me on discord at ChronicCoder#1667

### Credits

* Our amazing discord community: https://discord.gg/ZummSXK
