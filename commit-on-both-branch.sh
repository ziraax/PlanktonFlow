#!/bin/bash

git checkout development 
git add .
git commit -m "$1"
git push origin development 

git checkout main 
git merge development 
git push origin main 

git checkout development 