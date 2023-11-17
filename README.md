# anodet-anvil
An Anvil + Flask webapp for for computer vision anomlay detection with [Anodet](https://github.com/OpenAOI/anodet) framework.

## Requirements
- Python 3.10
- mkdocs, mkdocstrings, mkdocs-material
- anvil, anvil-uplink
- mccp


## Installation

Clone the repository
```
git clone https://github.com/rufshod/anodet-anvil.git
```

Clone the <a href="https://anvil.works/build#clone:JT6SCPFZFHLBPZ6K=SDLUIZFY3TF7DKYEMR2HJWDA|YZFAF3UQ2TEZDHTQ=DUSY6UWZ2WDCVXNUJYFEU6NW|5IMMJBKGCSO6YGHB=K2QYA32ANY6JM6L3SQVCNJWH">Anvil Works Project Source Code</a>

```bash
pip install mkdocs mdocstrings mdocs-material anvil anvil-uplink mccp
```

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


## What about uplink key?

Head over to Anvil Works to get your uplink key. Create anvil_key.json and place the uplink key in you root dir like this:
```json
{
    "Server_Uplink_Key": "server_XXXXXXXXXXXXXXXXXXXXXXXX-xxxxxxxxxxxxxxxxx"
}
```
