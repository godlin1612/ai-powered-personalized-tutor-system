import pandas as pd
import random
from faker import Faker
import numpy as np

fake = Faker()

# Sample values
countries = ['India', 'USA', 'UK']
states = {'India': ['Maharashtra', 'Karnataka'], 'USA': ['California', 'Texas'], 'UK': ['England', 'Scotland']}
cities = {'Maharashtra': ['Mumbai', 'Pune'], 'Karnataka': ['Bangalore', 'Mysore'],
          'California': ['Los Angeles', 'San Diego'], 'Texas': ['Austin', 'Dallas'],
          'England': ['London', 'Manchester'], 'Scotland': ['Glasgow', 'Edinburgh']}
occupations = ['Engineer', 'Teacher', 'Doctor', 'Farmer', 'Clerk', 'Business']
earning_class = ['Low', 'Middle', 'High']
levels = ['Beginner', 'Intermediate', 'Advanced']
courses = ['Math', 'Science', 'English']
materials = ['Video', 'PDF', 'Quiz']
material_levels = ['Basic', 'Medium', 'Hard']

data = []

for _ in range(1000):
    country = random.choice(countries)
    state = random.choice(states[country])
    city = random.choice(cities[state])
    
    age = random.randint(5, 18)
    iq = random.randint(80, 140)
    level_student = random.choice(levels)
    level_course = random.choice(levels)
    gender = random.choice(['Male', 'Female'])
    time_per_day = random.randint(10, 120)
    assessment_score = np.clip(random.gauss(70, 15), 0, 100)
    promoted = 'Yes' if assessment_score > 60 else 'No'

    data.append({
        "Name": fake.first_name(),
        "Age": age,
        "Gender": gender,
        "Country": country,
        "State": state,
        "City": city,
        "Parent Occupation": random.choice(occupations),
        "Earning Class": random.choice(earning_class),
        "Level of Student": level_student,
        "Level of Course": level_course,
        "Course Name": random.choice(courses),
        "Material Name": random.choice(materials),
        "Material Level": random.choice(material_levels),
        "Time per Day (min)": time_per_day,
        "Assessment Score": round(assessment_score, 2),
        "IQ of Student": iq,
        "Promoted": promoted
    })

df = pd.DataFrame(data)
df.to_csv("k12_students_data.csv", index=False)
print("✅ Dataset created: k12_students_data.csv")
