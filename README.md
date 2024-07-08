# Hand Mimicking System with Haptic Device

## Project Overview

This project aims to create a system where a learner can mimic hand gestures performed by an expert using a haptic device. Two laptops communicate via IP to enable real-time hand gesture recognition using webcam feeds processed with Mediapipe. Landmark points are extracted and used to compute angles for comparison, deriving error values which are then adjusted using PID control to manipulate the haptic device.

## Installation and Setup

### Prerequisites

- Python 3.x or higher
- Required Python libraries (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/universeflower/hapticinteraction.git
   cd hapticinteraction

2. install required libraries:
   ```bash
   pip install -r requirements.txt
## Usage
1. Run the project:
   python main.py
2. Set up IP communication between the two laptops and connect the webcam.

## Key Features
- Recognition of expert hand gestures and landmark extraction
- Calculation of angles from extracted landmarks to determine error values
- Adjustment of haptic device using PID control

## Project Structure
hapticinteraction/
├── main.py
├── README.md
├── requirements.txt
├── webcam.py
└── mediapipe_utils.py

## video
[![Server Demo](https://img.youtube.com/vi/burq02zPhr0/0.jpg)](https://youtu.be/burq02zPhr0)

[![Client Demo](https://img.youtube.com/vi/BlLi7VD5V_g/0.jpg)](https://youtu.be/BlLi7VD5V_g)

## result graph
![image](https://github.com/universeflower/hapticinteraction/assets/132273821/7e3b4318-03df-4cc1-ad62-65fd25c3c5b8)


