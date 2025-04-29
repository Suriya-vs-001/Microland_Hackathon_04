import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

nltk.download('punkt')
nltk.download('wordnet')

response_patterns = {
    "greeting": [
        "Hello! How can I assist you today?",
        "Hi there! How can I help with your healthcare needs?",
        "Greetings! What can I do for you?",
        "Hey! How's it going? How can I help?",
        "Hi! Need any assistance?",
        "Hello! Ready to help with your health queries!",
        "Good day! Let me know how I can assist.",
        "Hello! How can I support you today?",
    ],
    "how_are_you": [
        "I'm here to assist you with your questions!",
        "Thank you for asking! I’m here to help you.",
        "I'm ready to help! Let me know what you need.",
        "I'm a virtual assistant, so I don’t have feelings, but I'm ready to help!",
        "I’m great! I hope you are well too. How can I assist?",
        "I’m here and ready to help. How can I assist you?",
    ],
    "fever": [
        "For a fever, rest and drink lots of fluids. Monitor your temperature and consult a doctor if it’s high.",
        "Fever might indicate an infection. Stay hydrated and get some rest.",
        "If you have a fever, try to keep cool, stay hydrated, and rest. Consult a doctor if it persists.",
        "A fever may mean your body is fighting an infection. Drink fluids and rest. Seek medical help if needed.",
        "Fever can sometimes be managed with acetaminophen or ibuprofen, but consult a doctor if unsure.",
    ],
    "cold": [
        "A cold can be uncomfortable. Rest, stay hydrated, and consider over-the-counter medication if needed.",
        "For cold symptoms, drink warm fluids and rest. Over-the-counter decongestants may help.",
        "Coughing and sneezing can be signs of a cold. Keep warm, drink fluids, and get plenty of rest.",
        "A cold is often viral. Stay hydrated, rest, and let your body heal.",
        "Try honey with warm water for a sore throat, and steam inhalation for congestion.",
    ],
    "headache": [
        "A headache could be due to stress or dehydration. Drink water and rest if possible.",
        "Headaches might come from tension or lack of sleep. Try resting in a dark room.",
        "Try deep breathing for a headache, or a cool cloth on your forehead.",
        "Over-the-counter pain relievers may help with headaches, but check with a doctor if they persist.",
        "Stress-relieving activities like meditation or a warm shower may help with headaches.",
    ],
    "stomachache": [
        "Stomach aches can result from diet or stress. Try drinking warm water or herbal tea.",
        "Avoid spicy and heavy foods if you have stomach pain. Light meals and rest may help.",
        "Stomach discomfort may be eased with a warm compress or gentle stretching.",
        "If you have stomach pain, avoid alcohol and caffeine and try to rest.",
        "For ongoing stomach pain, consult a healthcare provider for guidance.",
    ],
    "sore_throat": [
        "A sore throat may improve with warm saltwater gargles and staying hydrated.",
        "Drinking warm fluids like tea with honey can soothe a sore throat.",
        "Resting your voice and using lozenges may help alleviate throat pain.",
        "Humidifiers can add moisture to the air, which might relieve a sore throat.",
        "If your throat pain persists, consult a healthcare provider for further advice.",
    ],
    "fatigue": [
        "Fatigue might come from lack of sleep or stress. Ensure you rest and hydrate.",
        "If you're often tired, check if you're getting enough sleep or managing stress effectively.",
        "Sometimes, fatigue can be due to low iron levels; consult a doctor if it continues.",
        "Balanced nutrition, hydration, and proper rest are key for reducing fatigue.",
        "Persistent fatigue might need medical attention; don't hesitate to consult a healthcare provider.",
    ],
    "stress": [
        "Managing stress is important. Breathing exercises, journaling, or talking to a friend can help.",
        "Stress can affect health. Try relaxation techniques or take some time for hobbies you enjoy.",
        "Deep breathing, meditation, or stretching exercises may reduce stress.",
        "Taking breaks, even short ones, during a busy day can help relieve stress.",
        "Talking about your stressors with a counselor or trusted person may offer relief.",
    ],
    "anxiety": [
        "Anxiety can be overwhelming. Breathing exercises and grounding techniques may help.",
        "If you're feeling anxious, consider mindfulness or meditation. It can bring a sense of calm.",
        "It’s okay to feel anxious sometimes. Speak with someone you trust if it helps.",
        "Grounding techniques like focusing on physical sensations can help reduce anxiety.",
        "Counseling or therapy can be very effective for managing anxiety.",
    ],
    "depression": [
        "Depression can feel isolating. Reach out to someone you trust, or consider speaking with a counselor.",
        "Remember, you're not alone. Talking to someone can help, and professional support is available.",
        "Taking small steps, like a walk outside, can sometimes help ease depressive feelings.",
        "Self-care is important; take time to rest, and don’t hesitate to ask for support if you need it.",
        "Mental health professionals can provide valuable support if you're feeling down.",
    ],
    "leg_sprain": [
        "For a sprained leg, try the R.I.C.E. method: Rest, Ice, Compression, and Elevation.",
        "If you have a leg sprain, avoid putting weight on it and use an ice pack to reduce swelling.",
        "Compression wraps and elevating your leg can help reduce sprain swelling.",
        "Avoid strenuous activity with a leg sprain and allow time for healing.",
        "If pain from a leg sprain persists, consult a doctor for further advice.",
    ],
    "asthma": [
        "For asthma, ensure you have access to your inhaler and avoid triggers like smoke or allergens.",
        "Asthma can flare up with allergens. Keep away from dust, smoke, and cold air when possible.",
        "A peak flow meter can help track your asthma symptoms. Consult a doctor if symptoms increase.",
        "Breathing exercises can sometimes help manage asthma symptoms. However, consult a healthcare provider for guidance.",
        "If you have asthma, stay hydrated, take medications as prescribed, and avoid known triggers.",
    ],
    "mental_health_tips": [
        "Taking care of mental health includes regular sleep, staying active, and connecting with loved ones.",
        "Engage in activities you enjoy to improve mental well-being.",
        "Practice gratitude and journaling to improve self-awareness and well-being.",
        "Taking a few moments daily to meditate or relax can improve mental health.",
        "Physical exercise, even a short walk, can help improve mood and mental clarity.",
    ],
    "thanks": [
        "You're welcome! Feel free to reach out if you have more questions.",
        "Happy to help! Let me know if you have other questions.",
        "No problem! Stay safe and take care.",
        "You're welcome! Remember, I’m here if you need more help.",
        "Glad to be of assistance!",
    ],
    "goodbye": [
        "Goodbye! Take care!",
        "See you later! Stay safe!",
        "Goodbye! Feel free to come back if you need more help.",
        "Take care! Wishing you good health.",
        "Goodbye! Don’t hesitate to reach out if you need advice again.",
    ],
    "back_pain": [
        "Back pain can often be eased with proper posture and gentle stretching.",
        "Try applying heat or cold to the affected area, and avoid heavy lifting.",
        "If you have persistent back pain, consider visiting a doctor for further evaluation.",
        "Ensure you’re maintaining a good posture throughout the day to prevent back pain.",
        "Gentle exercises and stretching might help alleviate back discomfort. Make sure not to overstrain."
    ],
    "cough": [
        "A persistent cough can be caused by allergies or a cold. Consider drinking warm tea with honey.",
        "Coughing can also be a sign of respiratory infections or irritation. Keep hydrated and rest.",
        "If you have a cough with mucus or blood, it’s best to consult a healthcare provider.",
        "A dry cough might improve with warm liquids and throat lozenges. If it persists, see a doctor.",
        "Use a humidifier to moisten the air if your cough is caused by dry air or a sore throat."
    ],
    "diabetes": [
        "Managing diabetes involves controlling blood sugar levels through diet, exercise, and medication.",
        "If you have diabetes, it’s important to monitor your blood sugar levels regularly.",
        "Eating a balanced diet with low sugar and carbohydrates can help control diabetes.",
        "Exercise is a great way to keep blood sugar levels in check if you have diabetes.",
        "Consult with a doctor to develop a personalized diabetes management plan, including medication if needed."
    ],
    "high_blood_pressure": [
        "To manage high blood pressure, reduce your salt intake, exercise regularly, and maintain a healthy weight.",
        "Regular monitoring of your blood pressure is essential for managing hypertension.",
        "If you have high blood pressure, try stress-relief techniques like yoga or meditation.",
        "Limit alcohol consumption and avoid smoking to help reduce high blood pressure.",
        "Consult with a healthcare provider to determine the best treatment for managing hypertension."
    ],
    "insomnia": [
        "For better sleep, establish a consistent bedtime routine and avoid screens before bed.",
        "Avoid caffeine or heavy meals in the evening to improve sleep quality.",
        "Relaxation techniques such as deep breathing or meditation can help with insomnia.",
        "If you have persistent insomnia, consider talking to a healthcare provider about possible treatments.",
        "Creating a calm, quiet sleeping environment may also help with insomnia."
    ],
    "allergy": [
        "If you have allergies, try avoiding known triggers, such as pollen, dust, or pet dander.",
        "Over-the-counter antihistamines may help with symptoms of allergies like sneezing and itchy eyes.",
        "If your allergies are severe, consult a doctor for other possible treatments.",
        "Consider using air purifiers and keeping windows closed to minimize allergy triggers in your home.",
        "Staying hydrated and using saline nasal sprays can help relieve some allergy symptoms."
    ],
    "heart_disease": [
        "Heart disease can often be prevented by eating a balanced diet, exercising regularly, and avoiding smoking.",
        "If you have heart disease, medications and lifestyle changes may help manage symptoms.",
        "Monitoring your cholesterol levels and blood pressure is crucial for preventing heart disease.",
        "Consult with your healthcare provider to develop a plan to manage your heart health effectively.",
        "Regular exercise, a healthy diet, and stress management are essential for heart health."
    ],
    "arthritis": [
        "Arthritis can cause joint pain and stiffness. Over-the-counter anti-inflammatory medications may help.",
        "Exercise, such as swimming or walking, can help keep joints flexible and reduce arthritis pain.",
        "Applying heat or cold to the affected area can provide relief from arthritis symptoms.",
        "If arthritis pain becomes unmanageable, consult a healthcare provider for treatment options.",
        "Try to maintain a healthy weight to reduce pressure on your joints, especially for those with osteoarthritis."
    ],
    "pregnancy": [
        "During pregnancy, it’s important to attend regular prenatal check-ups and follow your doctor’s advice.",
        "Eat a balanced diet rich in nutrients, including folic acid, iron, and calcium, during pregnancy.",
        "Make sure to stay hydrated and get enough rest throughout your pregnancy.",
        "If you're pregnant, avoid smoking, drinking alcohol, or taking unapproved medications.",
        "Consult your doctor for guidance on exercise and activities during pregnancy."
    ],
    "cholesterol": [
        "To manage high cholesterol, eat a diet low in saturated fats and cholesterol.",
        "Regular exercise and weight management can help reduce high cholesterol levels.",
        "If you have high cholesterol, your doctor may recommend medications to lower it.",
        "Avoid foods high in trans fats and opt for healthier fat sources, such as olive oil and nuts.",
        "Monitoring cholesterol levels regularly is crucial for heart health."
    ],
    "stroke": [
        "A stroke can cause sudden symptoms like numbness, confusion, or difficulty speaking. Immediate medical attention is crucial.",
        "If you or someone else shows signs of a stroke, seek emergency care immediately.",
        "Prevention of stroke includes controlling high blood pressure, maintaining a healthy diet, and not smoking.",
        "Rehabilitation and therapy may be needed following a stroke for physical or speech impairments.",
        "Consult with a healthcare provider for stroke prevention strategies and early signs to watch for."
    ],
    "kidney_disease": [
        "Kidney disease may cause symptoms like swelling, fatigue, and difficulty urinating. Early detection is important.",
        "Keep your blood pressure and blood sugar levels in check to prevent kidney disease.",
        "If you have kidney disease, your doctor may recommend dialysis or other treatments.",
        "Maintaining a healthy diet, staying hydrated, and avoiding smoking can help prevent kidney problems.",
        "Consult a healthcare provider for regular kidney function tests, especially if you have risk factors."
    ],
    "ulcer": [
        "Stomach ulcers can be caused by stress, medication, or infection. Consult a doctor for treatment.",
        "To help with ulcer pain, avoid spicy foods, alcohol, and caffeine.",
        "Certain medications like proton pump inhibitors may be prescribed to treat ulcers.",
        "If you have ulcers, it’s important to avoid NSAIDs and other irritating substances.",
        "Consult a doctor for a proper diagnosis and treatment plan if you suspect you have an ulcer."
    ],
    "back_injury": [
        "If you’ve injured your back, avoid heavy lifting and try to rest.",
        "Apply ice or heat to the affected area for 20 minutes at a time to alleviate pain.",
        "Gentle stretches and exercises can help strengthen your back muscles after an injury.",
        "If back pain persists, consult a healthcare provider for a treatment plan.",
        "Consider using ergonomic furniture and maintaining proper posture to prevent back injuries."
    ],
    "migraine": [
        "Migraines often cause severe pain, nausea, and sensitivity to light. Resting in a dark, quiet room may help.",
        "Over-the-counter pain relievers or migraine medications may provide relief.",
        "Try relaxation techniques, such as deep breathing or yoga, to reduce migraine frequency.",
        "Avoid migraine triggers like certain foods, stress, or bright lights when possible.",
        "If migraines are frequent, consult a doctor for other possible treatments or medication."
    ],
    "food_intolerance": [
        "Food intolerances can cause symptoms like bloating, gas, and stomach cramps. Identify and avoid trigger foods.",
        "If you have a food intolerance, try eliminating specific foods from your diet and see if symptoms improve.",
        "Lactose intolerance can be managed with lactase supplements or lactose-free dairy products.",
        "Gluten intolerance requires a strict gluten-free diet to manage symptoms.",
        "Consult a dietitian to help manage food intolerances and maintain a balanced diet."
    ],
    "hormonal_imbalance": [
        "Hormonal imbalances can cause symptoms like fatigue, weight gain, and mood swings.",
        "Diet, exercise, and proper sleep can help regulate hormones and reduce symptoms.",
        "If you suspect a hormonal imbalance, consult a doctor for blood tests and treatment options.",
        "Herbal remedies and supplements may help with hormone regulation, but speak with a doctor first.",
        "For persistent hormonal issues, hormone therapy or other medical interventions might be necessary."
    ],
    "pneumonia": [
        "Pneumonia can cause symptoms like cough, fever, chills, and shortness of breath. Seek medical attention immediately.",
        "Rest, stay hydrated, and follow prescribed antibiotics if you have bacterial pneumonia.",
        "Pneumonia may require hospitalization for severe cases, especially if you have breathing difficulties.",
        "If you have pneumonia, avoid smoking and ensure you’re getting enough rest to help your body recover.",
        "Vaccines are available to help prevent certain types of pneumonia. Talk to your doctor about your options."
    ],
    "feeling_alone": [
        "I'm really sorry to hear that you're feeling alone. Remember, it's okay to feel this way sometimes.",
        "You don’t have to go through it alone. It’s important to talk to someone you trust or seek support.",
        "Feeling alone can be tough, but it's just a temporary feeling. You can reach out to friends or family for comfort.",
        "Sometimes, all we need is a little reminder that we matter. You're important, and you're not alone in this.",
        "It’s okay to feel alone sometimes, but remember that reaching out for support can help you feel more connected."
    ],
    
    "motivational_tips": [
        "Motivation can be tough to find sometimes, but start with small, manageable goals. Celebrate your wins, no matter how small!",
        "To stay motivated, try focusing on your ‘why’ — what’s the reason behind what you’re doing?",
        "When motivation is low, it can help to break tasks into smaller, more manageable steps. You'll feel a sense of accomplishment along the way.",
        "Remember, motivation comes in waves. Some days will be harder than others, but keep pushing forward, one step at a time.",
        "Try to set positive routines that make you feel energized. Also, don't forget to reward yourself when you complete tasks!"
    ],
    
    "self_care": [
        "Self-care is so important! Take some time for yourself, even if it's just a few minutes to relax and breathe deeply.",
        "Consider taking a walk outside or listening to your favorite music — it can help lift your mood and clear your mind.",
        "Self-care isn't selfish. It’s essential for your mental health. Try practicing mindfulness or a hobby that makes you feel good.",
        "If you're feeling down, sometimes doing something small that makes you feel good, like reading or journaling, can help boost your spirits.",
        "Taking care of your mental health is just as important as physical health. Never feel guilty about needing a break."
    ],
    
    "encouragement": [
        "You’re doing better than you think. Keep going, and don’t be too hard on yourself.",
        "You’ve faced tough moments before, and you’ve made it through. You’re strong enough to keep going.",
        "Even when things feel difficult, remember that you are making progress. Take it one step at a time.",
        "Stay focused on your goals and take things one day at a time. You’ve got this!",
        "Believe in yourself. Every challenge is an opportunity to grow stronger and wiser."
    ],
    "breathing": [
        "Breathing trouble can be caused by a variety of factors. If you're feeling short of breath, it's important to stay calm.",
        "If you're having difficulty breathing, try to find a comfortable position and breathe slowly. If the problem persists, it's best to seek medical attention.",
        "Breathing issues can be serious. If you're experiencing severe shortness of breath, it's important to call a healthcare provider immediately.",
        "If you're struggling to breathe, it could be a sign of asthma, allergies, or another underlying condition. Have you been exposed to any allergens or irritants?",
        "Breathing problems might be related to asthma, allergies, or infections. It's important to track any additional symptoms and consult with a doctor if necessary."
    ],

    "asthma": [
        "Asthma can cause shortness of breath, wheezing, and coughing. If you're experiencing an asthma attack, using your inhaler can help.",
        "If you're having difficulty breathing, make sure you're in a well-ventilated area and try to use your prescribed inhaler or medication.",
        "Asthma symptoms can be triggered by various factors like allergies, cold air, or exercise. If you're struggling to breathe, it may be a good idea to use a bronchodilator and consult a healthcare provider."
    ],

    "allergies_breathing": [
        "If you're having trouble breathing due to allergies, antihistamines might help. It's also important to avoid allergens like dust or pollen.",
        "Breathing trouble due to allergies can often be relieved by taking allergy medication. Be sure to check with your healthcare provider for the right treatment.",
        "If you have seasonal allergies or exposure to dust or pet dander, it could be triggering your breathing issues. Try using an air purifier or wearing a mask."
    ],

    "chest_pain": [
        "Chest pain can be caused by many things, ranging from anxiety to a heart-related issue. If the pain is severe or accompanied by shortness of breath, you should seek medical attention immediately.",
        "If you're experiencing chest pain, it's important to assess if it's due to stress, muscle strain, or something more serious like a heart problem. Seek medical help if you're uncertain.",
        "Chest pain with difficulty breathing may indicate a serious condition. Please consult a healthcare professional if you're feeling unwell."
    ],

    "shortness_of_breath": [
        "Shortness of breath can occur due to physical exertion, allergies, or respiratory infections. If you're at rest and still experiencing difficulty, it’s important to consult a healthcare provider.",
        "Difficulty breathing can be caused by conditions like asthma, pneumonia, or even anxiety. Take deep, slow breaths and seek medical advice if it persists.",
        "If you're experiencing shortness of breath, rest and avoid physical exertion. If this happens frequently or is severe, see a doctor."
    ],

    "pneumonia": [
        "Pneumonia can cause difficulty breathing, chest pain, and fever. If you suspect you have pneumonia, it's important to consult a healthcare provider for proper treatment.",
        "If you're having trouble breathing and suspect pneumonia, it's important to get medical attention immediately. Pneumonia can be treated with antibiotics if bacterial, or antiviral if viral.",
        "Difficulty breathing, along with fever or a cough, could be signs of pneumonia. Please seek medical advice if you suspect you have this condition."
    ],

    "COPD": [
        "Chronic Obstructive Pulmonary Disease (COPD) causes difficulty breathing, coughing, and wheezing. If you have COPD, it's essential to follow your doctor’s prescribed treatment plan.",
        "COPD can lead to progressive difficulty in breathing. If you're having trouble breathing, make sure you're following your medication plan and consult your doctor if symptoms worsen.",
        "If you’re experiencing difficulty breathing due to COPD, it’s important to stay on top of your medications and consider seeking professional help if you're feeling overwhelmed."
    ],

    "anxiety_breathing": [
        "Anxiety can cause difficulty breathing and a sense of tightness in the chest. Deep breathing exercises or mindfulness may help ease anxiety and improve your breathing.",
        "If your breathing issues are linked to anxiety, try calming techniques such as deep breathing, meditation, or gentle stretching. If this is a recurring issue, you might want to consult with a therapist.",
        "When anxiety causes breathing difficulties, grounding exercises like deep breathing or counting can be helpful. It's important to recognize if anxiety is the cause and address it with self-care."
    ],

    "stress_breathing": [
        "Stress can lead to shallow, rapid breathing. Try to focus on slow, deep breaths to calm yourself down. If stress continues to cause breathing problems, consider stress management techniques.",
        "If stress is causing difficulty breathing, try relaxation techniques such as progressive muscle relaxation or deep breathing. Speaking to a healthcare provider may also be beneficial if stress becomes overwhelming."
    ],

    "default": [
        "I'm sorry, I didn't quite understand that. Can you please rephrase?",
        "I'm here to help. Could you share a bit more about how you're feeling?",
        "Could you clarify that a bit? I'll do my best to help with any questions you have.",
        "I’m not sure I understand. Could you ask something else or explain a little more?"
    ]

}


def classify_intent(user_input):
    user_input = user_input.lower()
    keywords = {
        "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
        "how_are_you": ["how are you", "how's it going", "how have you been"],
        "fever": ["fever", "temperature", "hot", "feverish", "high temperature"],
        "cold": ["cold", "cough", "sneezing", "runny nose", "nasal congestion"],
        "headache": ["headache", "head pain", "migraine", "head hurts"],
        "breathing_trouble": ["breathing trouble", "difficulty breathing", "shortness of breath", "can't breathe properly", "breathless"],
        "stomachache": ["stomach ache", "stomach pain", "abdominal pain", "stomach hurt"],
        "sore_throat": ["sore throat", "throat pain", "dry throat", "scratchy throat"],
        "fatigue": ["fatigue", "tired", "exhausted", "low energy"],
        "stress": ["stress", "stressed", "overwhelmed", "burned out"],
        "anxiety": ["anxiety", "anxious", "nervous", "panic"],
        "depression": ["depression", "depressed", "sad", "feeling down", "hopeless"],
        "leg_sprain": ["leg sprain", "sprained leg", "hurt my leg", "twisted my ankle"],
        "asthma": ["asthma", "shortness of breath", "wheezing", "inhaler", "breathing problems"],
        "mental_health_tips": ["mental health", "well-being", "mindfulness", "self-care"],
        "thanks": ["thank you", "thanks", "appreciate", "grateful"],
        "goodbye": ["goodbye", "bye", "see you", "farewell"],
        "feeling_alone": ["feel alone", "feeling lonely", "feeling isolated", "i feel alone", "alone today"],
        "motivational_tips": ["motivated", "motivation", "how to feel motivated", "lack of motivation", "i need motivation"],
        "self_care": ["self-care", "take care of myself", "relax", "stress relief", "pamper myself"],
        "encouragement": ["i can’t do this", "i feel like giving up", "i’m stuck", "i feel overwhelmed", "i’m struggling"],
        "back_pain": ["back pain", "lower back pain", "upper back pain", "back hurts"],
        "cough": ["cough", "dry cough", "productive cough", "persistent cough"],
        "diabetes": ["diabetes", "blood sugar", "glucose", "insulin"],
        "high_blood_pressure": ["high blood pressure", "hypertension", "high bp"],
        "insomnia": ["insomnia", "can't sleep", "sleep issues", "sleeping problems"],
        "allergy": ["allergy", "pollen", "sneezing", "itchy eyes", "dust"],
        "heart_disease": ["heart disease", "heart attack", "chest pain", "cardiac issues"],
        "arthritis": ["arthritis", "joint pain", "rheumatoid arthritis", "osteoarthritis"],
        "pregnancy": ["pregnancy", "pregnant", "expecting", "baby"],
        "cholesterol": ["cholesterol", "lipid", "cholesterol levels", "high cholesterol"],
        "stroke": ["stroke", "brain attack", "slurred speech", "weakness on one side"],
        "kidney_disease": ["kidney disease", "renal disease", "dialysis", "kidney failure"],
        "ulcer": ["ulcer", "stomach ulcer", "gastric ulcer", "peptic ulcer"],
        "back_injury": ["back injury", "spinal injury", "hernia", "back strain"],
        "migraine": ["migraine", "headache", "severe headache", "throbbing headache"],
        "food_intolerance": ["food intolerance", "lactose", "gluten", "food allergy", "milk intolerance"],
        "hormonal_imbalance": ["hormonal imbalance", "hormones", "thyroid", "hormonal changes"],
        "pneumonia": ["pneumonia", "lung infection", "chest infection", "breathing difficulties"]
    }
    
    for intent, terms in keywords.items():
        if any(term in user_input for term in terms):
            return intent
    return "default"


def generate_response(intent):
    responses = response_patterns.get(intent, response_patterns["default"])
    return random.choice(responses)

def get_response(user_input):
    intent = classify_intent(user_input)
    response = generate_response(intent)
    return response

if __name__ == "__main__":
    print("Healthcare Chatbot: Hello! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Healthcare Chatbot: Take care! Goodbye!")
            break
        response = get_response(user_input)
        print(f"Healthcare Chatbot: {response}")
