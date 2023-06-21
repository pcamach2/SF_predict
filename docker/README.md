Clone the micaopen/sf_prediction github repo for build context

```
git clone https://github.com/MICA-MNI/micaopen.git
docker build -t pyconnpredict:1.0.0 .
```

To build Singularity container:
```
git clone https://github.com/MICA-MNI/micaopen.git
docker build -t local/pyconnpredict:1.0.0 . && docker push local/pyconnpredict:1.0.0
singularity build pyconnpredict-v1.0.0.sif docker-daemon://local/pyconnpredict:1.0.0
```
