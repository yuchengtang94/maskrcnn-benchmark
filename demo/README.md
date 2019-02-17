##  Jupyter notebook demo and service

You can have a look at the text detection demo at these notebook.
```
text_demo.ipynb
text_detection_visualize.ipynb
```

The service is using flask & gevent framework to enable paralell processing.

It will be run on http://0.0.0.0:8000/predict

Run the predict service by

```
python service.py
```

You can test it in

```
service_test.ipynb
```


