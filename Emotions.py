# a. Define Emotions
emotions = [
    "Happiness",   # Recognizing a positive and content emotion.
    "Sadness",     # Identifying expressions associated with sadness or distress.
    "Anger",       # Detecting signs of anger or frustration.
    "Surprise",    # Capturing expressions of surprise or astonishment.
    "Disgust",     # Recognizing expressions of dislike or repulsion.
    "Fear",        # Identifying facial features associated with fear or anxiety.
    "Neutral"      # Recognizing a neutral expression, devoid of strong emotion.
]

# b. Choose Application Area
application_areas = [
    "User Experience Design",
    "Human-Computer Interaction",
    "Emotion-Aware Applications",
    "Marketing and Advertising",
    "Healthcare and Mental Health",
    "Security and Surveillance",
    "Education"
]

# Documentation

print("Defined Emotions:")
for emotion in emotions:
    print("- {}".format(emotion))

print("\nChosen Application Areas:")
for area in application_areas:
    print("- {}".format(area))
