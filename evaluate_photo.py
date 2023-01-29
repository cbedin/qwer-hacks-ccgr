
#for single image
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)

#if it is most likely that the picture is a car, return 1
if np.argmax(predictions[0]) = 2:
    return 1
else:
    print('Not police')