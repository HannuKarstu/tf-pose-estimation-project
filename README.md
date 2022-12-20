# Pose Estimation AI project

Kajaani UAS practice project

2022

## Notebooks

### Development and requirements

1. Clone repository and install Docker

2. (optional) Create Python virtual environment: https://docs.python.org/3/library/venv.html

3. Install dependencies from pose_estimation_tests.ipynb

4. Download MPII Human Pose Dataset http://human-pose.mpi-inf.mpg.de/#download

5. Download the dataset and place images in "pose_estimation_benchmark_images" above this repository's folder.

6. Download converted annotations, extract archive and place the file 'mpii_val.json' in the same "pose_estimation_benchmark_images" folder as above: https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar (more info: https://mmpose.readthedocs.io/en/latest/tasks/2d_body_keypoint.html)

### Pose estimation tests

Notebook for running single pose estimation models with a sample image and overlaying the keypoints to original image. Also test 

### Benchmark

Benchmark requires MPII human pose dataset and converted annotations to function.

## Web application

### Development

Install and use npm package 'live-server' for automatically reloading web application:

Install live-server globally:

`npm install live-server -g`

Open web application's folder:

`cd web_app`

Start live-server:

`live-server $PWD`

### Deployment

Run the web application in Docker container:

`docker run --name ml_nginx -d -p 8080:80 -v /ai_and_web_applications_project/web_app/:/usr/share/nginx/html/ nginx`

## Authors

Hannu Karstu

Marked parts by:

- Pekka Huttunen
- Tensorflow development team

## License

Open for all use
