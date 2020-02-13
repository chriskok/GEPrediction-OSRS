# OSRS Grand Exchange Price Prediction

Start-to-end project where we attempt to harness the power of machine learning to predict Old-school Runescape Grand Exchange prices.

## Getting started
The journey so far has been documented in a series of Youtube videos found here: 
1. [Part 1](https://youtu.be/D5TmBcpgm7k) - Setup and initial trial
2. [Part 2](https://youtu.be/U453OSC8dkc) - Data Collection
3. Part 3 (TBD) - Feature Engineering and Selection

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

### Running the jupyter notebooks

Run the following command in the current directory and run the notebooks from there: 
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
