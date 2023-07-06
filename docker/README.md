Clone the micaopen/sf_prediction github repo for build context

If you want to save the scores from training data, pull from my [fork](https://github.com/pcamach2/micaopen.git)

```
git clone https://github.com/pcamach2/micaopen.git
# move sf_prediction folder
mv micaopen/sf_prediction ./sf_prediction
# remove unused portions of micaopen repo
rm -rf micaopen
docker build -t pyconnpredict:1.0.0 .
```

To build Singularity container:
```
git clone https://github.com/pcamach2/micaopen.git
# move sf_prediction folder
mv micaopen/sf_prediction ./sf_prediction
# remove unused portions of micaopen repo
rm -rf micaopen
docker build -t local/pyconnpredict:1.0.0 . && docker push local/pyconnpredict:1.0.0
singularity build pyconnpredict-v1.0.0.sif docker-daemon://local/pyconnpredict:1.0.0
```
