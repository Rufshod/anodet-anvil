# Welcome to Anodet-Anvil

Anodet-Anvil is a web application combining Anvil and Flask for computer vision anomaly detection using the Anodet framework.

## Getting Started

### Installation

Clone the Anodet-Anvil repository:

```bash
git clone https://github.com/rufshod/anodet-anvil.git
```
Also, clone the [Anvil Works Project Source Code](https://anodet.anvil.app).
# Setting Up Your Environment

Create a virtual environment and install the necessary requirements (details to be added).
# Warehouse Setup

The data warehouse structure is crucial for the proper functioning of the application. is created automatically when running the app:

```scss
data_warehouse
 ┣ dataset
 ┃ ┗ your_object_name
 ┃   ┣ test
 ┃   ┃ ┃ good 
 ┃   ┃ ┗ Anomaly (test images here)
 ┃   ┗ train
 ┃     ┗ good (angle directories here)
 ┣ distributions (object_name directories here)
 ┗ plots (prediction plots directories here)
```
If installed correctly, [MCCP](https://github.com/wlinds/mccp) should set up the structure automatically.
# Uplink Key

Obtain your server uplink key from Anvil Works and add it to a JSON file in your root directory:

```json
{
    "Server_Uplink_Key": "server_XXXXXXXXXXXXXXXXXXXXXXXX-xxxxxxxxxxxxxxxxx"
}
```
# Documentation

This documentation will guide you through the features and functionalities of Anodet-Anvil. Explore the sections to learn more about how to use the application effectively for anomaly detection in computer vision tasks.