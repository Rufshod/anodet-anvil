# anodet-anvil
An Anvil + Flask webapp for for computer vision anomlay detection with [Anodet](https://github.com/OpenAOI/anodet) framework.

## Installation

Clone the repository
```
git clone https://github.com/rufshod/anodet-anvil.git
```

## Create a venv
TODO add requirements etc.

## Warehouse setup

If installed correctly, [MCCP](https//github.com/wlinds/mccp), should setup the structure automatically. But if you for some reason need to create or modify it it should look like this:

```
data_warehouse
 ┣ dataset
 ┃ ┗ your_object_name
 ┃   ┣ test
 ┃   ┃ ┃ good 
 ┃   ┃ ┗ Anomaly (<- test images here)
 ┃   ┗ train
 ┃     ┗ good (<- angle dirs here)
 ┣ distributions <- object_name dirs here
 ┗ plots (<- angle dirs here)
```