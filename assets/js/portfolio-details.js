// Project data
const projectData = {
    'atlworks.ai': {
        title: 'Atlworks.ai',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/Atlworks.ai',
        media: [{ 'type': 'image', 'src': 'images/project/atlworks.ai.png', 'alt': 'Atlworks.ai Interface' }],
        description: `Welcome to ALT Works AI, a streamlined dashboard designed to showcase the innovative projects of AlphaTech Logics.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Welcome to ALT Works AI, a streamlined dashboard designed to showcase the innovative projects of AlphaTech Logics.</p>

        <h4>Features</h4>
        <p>Dynamic Data Fetching: Retrieves real-time project data using the GitHub API.
Interactive Dashboard: Displays projects in a responsive grid layout with modern card designs.
Icon-Based Navigation: Uses FontAwesome icons for intuitive navigation to GitHub and live demos.
Custom Tooltips: Each project card shows a short description on hover for quick insights.
Easy Demo URL Management: Manually map demo URLs for each project to ensure accurate linking.
</p>

        <h4>Development Stack & Skills</h4>
        <p>Streamlit
Python
GitHub API
FontAwesome</p>
    `
    },

    'news_summarizer_poc': {
        title: 'News-summarizer-poc',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/News-summarizer-poc',
        media: [{ 'type': 'image', 'src': 'images/project/news_summarizer_poc.png', 'alt': 'News-summarizer-poc Interface' }],
        description: `This project is a Streamlit-based application that fetches and analyzes news articles related to a specific stock ticker. It provides sentiment analysis, summarization, and sector/ticker extraction for each article.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This project is a Streamlit-based application that fetches and analyzes news articles related to a specific stock ticker. It provides sentiment analysis, summarization, and sector/ticker extraction for each article.</p>

        <h4>Features</h4>
        <p>Fetch news articles using the NewsAPI.
Perform sentiment analysis using FinBERT.
Summarize articles using BART.
Extract related stock tickers and sectors from the article content.</p>

        <h4>Development Stack & Skills</h4>
        <p>Python</p>
    `
    },

    'sentiment_analysis': {
        title: 'Sentiment-Analysis',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/Sentiment-Analysis',
        media: [{ 'type': 'image', 'src': 'images/project/sentiment_analysis.png', 'alt': 'Sentiment-Analysis Interface' }],
        description: `This application leverages a simple Recurrent Neural Network (RNN) to determine whether an IMDB movie review is positive or negative. Built with Streamlit, the app offers an intuitive and interactive user interface, allowing you to input your own reviews and receive instant sentiment predictions.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This application leverages a simple Recurrent Neural Network (RNN) to determine whether an IMDB movie review is positive or negative. Built with Streamlit, the app offers an intuitive and interactive user interface, allowing you to input your own reviews and receive instant sentiment predictions.</p>

        <h4>Features</h4>
        <p>🔍 Sentiment Analysis: Accurately classifies movie reviews as positive or negative.
🖥️ Interactive UI: Engage with a user-friendly Streamlit web app to input reviews and receive real-time predictions.
⚡ Pre-trained Model: Utilizes a pre-trained RNN model for swift and reliable sentiment analysis.
</p>

        <h4>Development Stack & Skills</h4>
        <p>Jupyter Notebook, Python and streamlit</p>
    `
    },

    'audio_recognition_ai_engine': {
        title: 'Audio-Recognition-AI-Engine',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/Audio-Recognition-AI-Engine',
        media: [{ 'type': 'image', 'src': 'images/project/audio_recognition_ai_engine.png', 'alt': 'Audio-Recognition-AI-Engine Interface' }],
        description: `initial commit for Audio-Recognition-AI-Engine`,
        fullDescription: `
        <h4>Description</h4>
        <p>initial commit for Audio-Recognition-AI-Engine</p>

        <h4>Features</h4>
        <p>nan</p>

        <h4>Development Stack & Skills</h4>
        <p>nan</p>
    `
    },

    'multimodal_rag': {
        title: 'MultiModal-RAG',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/MultiModal-RAG',
        media: [{ 'type': 'image', 'src': 'images/project/multimodal_rag.png', 'alt': 'MultiModal-RAG Interface' }],
        description: `This system enables you to effortlessly upload PDF documents and pose natural language queries. It seamlessly integrates a FastAPI-powered backend with a user-friendly frontend built using Streamlit.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This system enables you to effortlessly upload PDF documents and pose natural language queries. It seamlessly integrates a FastAPI-powered backend with a user-friendly frontend built using Streamlit.</p>

        <h4>Features</h4>
        <p>Document Submission: Easily send in your PDF documents for indexing.
Intelligent Q&A: Ask questions about your PDFs and receive context-rich, human-like answers.
User-Friendly UI: Enjoy a clean, intuitive interface that simplifies interaction.
Interactive Frontend: Benefit from an interactive Streamlit-based frontend for enhanced user experience.</p>

        <h4>Development Stack & Skills</h4>
        <p>nan</p>
    `
    },

    'qanda_system': {
        title: 'QandA-System',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/QandA-System',
        media: [{ 'type': 'image', 'src': 'images/project/qanda_system.png', 'alt': 'QandA-System Interface' }],
        description: `This is a simple yet powerful application built with Streamlit and Google Gemini LLM to interact with the generative AI model for answering questions in a conversational style.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This is a simple yet powerful application built with Streamlit and Google Gemini LLM to interact with the generative AI model for answering questions in a conversational style.</p>

        <h4>Features</h4>
        <p>Natural Language Processing with Google Gemini LLM.
 Real-time Q&A interaction.
 Session History to maintain conversation context.
 Error Handling to ensure smooth user experience.</p>

        <h4>Development Stack & Skills</h4>
        <p>nan</p>
    `
    },

    'unitest_generator': {
        title: 'UniTest-Generator',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/UniTest-Generator',
        media: [{ 'type': 'image', 'src': 'images/project/unitest_generator.png', 'alt': 'UniTest-Generator Interface' }],
        description: `A powerful test generation tool that uses AI to automatically generate test cases for both Java backend and Frontend (JavaScript/JSP) applications.`,
        fullDescription: `
        <h4>Description</h4>
        <p>A powerful test generation tool that uses AI to automatically generate test cases for both Java backend and Frontend (JavaScript/JSP) applications.</p>

        <h4>Features</h4>
        <p>Java Backend Testing

Automatic test case generation for Java classes
JUnit 5 test framework support
Mockito integration for mocking
JaCoCo coverage reporting
Support for both Maven and Gradle projects
Frontend Testing

JavaScript/TypeScript test generation
JSP test scenario generation
React, Vue, and Angular support
Jest test framework integration
Selenium for JSP testing
Coverage reporting with Istanbul/Jest</p>

        <h4>Development Stack & Skills</h4>
        <p>Python, java, </p>
    `
    },

    'garbage_sorting_image_classification': {
        title: 'garbage-sorting-image-classification',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/garbage-sorting-image-classification',
        media: [{ 'type': 'image', 'src': 'images/project/garbage_sorting_image_classification.png', 'alt': 'garbage-sorting-image-classification Interface' }],
        description: `This project is an AI-powered image classification web application that classifies uploaded waste images into one of four categories.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This project is an AI-powered image classification web application that classifies uploaded waste images into one of four categories.</p>

        <h4>Features</h4>
        <p>Image Classification: Predicts waste categories (METAL, ORGANIC, PAPER, PLASTIC).
Pretrained Model: MobileNetV2 pretrained on ImageNet, fine-tuned on a waste dataset.
Interactive Web App: Built using Streamlit for easy image uploads and predictions.</p>

        <h4>Development Stack & Skills</h4>
        <p>nan</p>
    `
    },

    'age_gender': {
        title: 'Retail Analytics POC: Age & Gender Detection',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'July 2025',
        url: 'https://github.com/alphatechlogics/Sense-of-Retail',
        media: [{ 'type': 'image', 'src': 'images/project/Age_gender.png', 'alt': 'age_gender' }],
        description: `This project is a real-time customer analytics system designed to monitor store activity and provide insights into customer demographics.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This project is a real-time customer analytics system designed to monitor store activity and provide insights into customer demographics. By combining YOLO for gender detection and DeepFace for age estimation, the system enables accurate profiling of customer visits across different store sections. A Streamlit dashboard was deployed within two weeks, delivering live data refresh and interactive visual analytics for quick decision-making</p>

        <h4>Features</h4>
        <p>Real-Time Tracking: Monitors and records customer visits across store sections.

Gender Detection: Achieves ~93% precision using a fine-tuned YOLO model.

Age Estimation: Provides age prediction with <5% error using DeepFace.

Visual Analytics Dashboard: Built with Streamlit, supports live data refresh and dynamic visualizations.</p>

        <h4>Development Stack & Skills</h4>
        <p>Python, DeepFace, YOLO</p>
    `
    },

    'text_classification': {
        title: 'Text-Classification',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/Text-Classification',
        media: [{ 'type': 'image', 'src': 'images/project/text_classification.png', 'alt': 'Text-Classification Interface' }],
        description: `This Natural Language Processing (NLP) application classifies BBC news articles into various genres such as Sports, Politics, Entertainment, Business, and Technology. Utilizing powerful models like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit)`,
        fullDescription: `
        <h4>Description</h4>
        <p>This Natural Language Processing (NLP) application classifies BBC news articles into various genres such as Sports, Politics, Entertainment, Business, and Technology. Utilizing powerful models like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit)</p>

        <h4>Features</h4>
        <p>Predicted Category: Displays the genre the article belongs to.
Prediction Score: Shows the confidence level of the prediction.
Prediction Plot: Visualizes the prediction scores across different genres.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit</p>
    `
    },

    'text_summarization': {
        title: 'Text-Summarization',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/Text-Summarization',
        media: [{ 'type': 'image', 'src': 'images/project/text_summarization.png', 'alt': 'Text-Summarization Interface' }],
        description: `A sleek and efficient web-based application for summarizing text and PDF documents, built with txtai and Streamlit.`,
        fullDescription: `
        <h4>Description</h4>
        <p>A sleek and efficient web-based application for summarizing text and PDF documents, built with txtai and Streamlit.</p>

        <h4>Features</h4>
        <p>Text Summarization: Generate concise and to-the-point summaries of any input text, regardless of length.
Powered by Cutting-Edge LLM: Utilizes the Gemma2-9b-It model via LangChain for accurate and context-aware summaries.
User-Friendly Interface: Simple, fast, and intuitive interface for seamless text summarization.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit</p>
    `
    },

    'doctor_ai': {
        title: 'Doctor-AI',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/Doctor-AI',
        media: [{ 'type': 'image', 'src': 'images/project/doctor_ai.png', 'alt': 'Doctor-AI Interface' }],
        description: `
This is an AI-powered Streamlit application designed to help analyze user-provided images and questions about skin conditions.`,
        fullDescription: `
        <h4>Description</h4>
        <p>
This is an AI-powered Streamlit application designed to help analyze user-provided images and questions about skin conditions.</p>

        <h4>Features</h4>
        <p>Image-based Diagnosis: Upload an image of a skin condition to receive an AI-powered diagnosis and treatment suggestions.
Interactive Chat: Ask follow-up questions based on the diagnosis or previous chat history.
Streamlit Frontend: A user-friendly interface for image uploads and interactive conversations.
FastAPI Backend: A REST API to integrate diagnosis and chat functionality into other applications.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit</p>
    `
    },

    'construcsafe': {
        title: 'ConstrucSafe',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/ConstrucSafe',
        media: [{ 'type': 'image', 'src': 'images/project/construcsafe.png', 'alt': 'ConstrucSafe Interface' }],
        description: `ConstrucSafe AI is a state-of-the-art application designed to enhance safety on construction sites through real-time object detection. Leveraging the power of YOLOv8, this tool detects and highlights safety equipment and potential hazards in images and videos, ensuring a safer working environment.`,
        fullDescription: `
        <h4>Description</h4>
        <p>ConstrucSafe AI is a state-of-the-art application designed to enhance safety on construction sites through real-time object detection. Leveraging the power of YOLOv8, this tool detects and highlights safety equipment and potential hazards in images and videos, ensuring a safer working environment.</p>

        <h4>Features</h4>
        <p>Image Detection: Upload images to identify safety gear and hazards.
Video Detection: Process and analyze videos to monitor safety compliance in real-time.
Customizable Thresholds: Adjust confidence levels to fine-tune detection accuracy.
Downloadable Results: Save processed images and videos with annotations for record-keeping.
User-Friendly Interface: Intuitive design built with Streamlit for seamless user experience.
</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit</p>
    `
    },

    'mathtutor': {
        title: 'MathTutor',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/MathTutor',
        media: [{ 'type': 'image', 'src': 'images/project/mathtutor.png', 'alt': 'MathTutor Interface' }],
        description: `Advanced AI Math Tutor is a Streamlit-based web application designed to help users learn and practice various mathematical concepts. It leverages AI to provide problem-solving assistance, practice questions, concept exploration, and more.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Advanced AI Math Tutor is a Streamlit-based web application designed to help users learn and practice various mathematical concepts. It leverages AI to provide problem-solving assistance, practice questions, concept exploration, and more.</p>

        <h4>Features</h4>
        <p>Problem Solver: Solve math problems step-by-step.
Practice Questions: Generate and practice custom questions.
Concept Explorer: Dive deep into mathematical concepts.
Formula Generator: Generate relevant formulas for selected topics.
Graph Visualizer: Visualize mathematical functions and analyze their properties.
Quiz: Take quizzes to test your knowledge.
Interactive Whiteboard: Draw and interpret mathematical expressions.
Virtual Math Manipulatives: Interactive tools for visual learning.
Study Plan Generator: Create personalized study plans.
Performance Analytics: Track your progress and performance.
AI Tutor Chat: Interact with an AI-powered math tutor.
Math Game Center: Engage with math-based games for fun learning.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit</p>
    `
    },

    'fraudulentdetection': {
        title: 'FraudulentDetection',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/FraudulentDetection',
        media: [{ 'type': 'image', 'src': 'images/project/fraudulentdetection.png', 'alt': 'FraudulentDetection Interface' }],
        description: `This project aims to build a machine learning model to detect fraudulent credit card transactions using an ensemble approach.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This project aims to build a machine learning model to detect fraudulent credit card transactions using an ensemble approach.</p>

        <h4>Features</h4>
        <p>nan</p>

        <h4>Development Stack & Skills</h4>
        <p>nan</p>
    `
    },

    'icumonitorscreenreader': {
        title: 'ICUMonitorScreenReader',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/ICUMonitorScreenReader',
        media: [{ 'type': 'image', 'src': 'images/project/icumonitorscreenreader.png', 'alt': 'ICUMonitorScreenReader Interface' }],
        description: `This app utilizes OpenAI's Vision API to analyze ICU monitor images, extract vital signs like heart rate (HR), blood pressure (BP), oxygen saturation (SpO2), respiratory rate (RR), temperature, and more.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This app utilizes OpenAI's Vision API to analyze ICU monitor images, extract vital signs like heart rate (HR), blood pressure (BP), oxygen saturation (SpO2), respiratory rate (RR), temperature, and more.</p>

        <h4>Features</h4>
        <p>Upload ICU Monitor Image: Upload an image from a patient monitor.
Image Analysis: The app uses OpenAI's Vision API to analyze the uploaded image and extract vital signs.
Data Extraction: The extracted vital signs such as HR, BP, SpO2, and more are parsed and displayed in a structured tabular format.
Streamlit Interface: The app provides a user-friendly interface where users can upload images and view the extracted data.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit</p>
    `
    },

    'plantleafdiseasedetection': {
        title: 'PlantLeafDiseaseDetection',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/PlantLeafDiseaseDetection',
        media: [{ 'type': 'image', 'src': 'images/project/plantleafdiseasedetection.png', 'alt': 'PlantLeafDiseaseDetection Interface' }],
        description: `This project is a Deep Learning-based Leaf Disease Detection System that utilizes transfer learning to identify and classify diseases in plant leaves. The app leverages a pre-trained model to provide high-accuracy predictions for 33 different types of leaf diseases.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This project is a Deep Learning-based Leaf Disease Detection System that utilizes transfer learning to identify and classify diseases in plant leaves. The app leverages a pre-trained model to provide high-accuracy predictions for 33 different types of leaf diseases.</p>

        <h4>Features</h4>
        <p>mage Upload: Upload a clear image of a leaf.
Disease Detection: Predicts the disease with confidence scores.
Interactive UI: Built with Streamlit for an engaging user experience.</p>

        <h4>Development Stack & Skills</h4>
        <p>python</p>
    `
    },

    'breastcancerdetection_': {
        title: 'BreastCancerDetection-',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/BreastCancerDetection-',
        media: [{ 'type': 'image', 'src': 'images/project/breastcancerdetection_.png', 'alt': 'BreastCancerDetection- Interface' }],
        description: `Welcome to the Breast Tumor Detection project! This project demonstrates to build, train, and deploy a deep learning model that detects breast tumors from mammogram images.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Welcome to the Breast Tumor Detection project! This project demonstrates to build, train, and deploy a deep learning model that detects breast tumors from mammogram images.</p>

        <h4>Features</h4>
        <p>nan</p>

        <h4>Development Stack & Skills</h4>
        <p>nan</p>
    `
    },

    'carnumberplatedetection': {
        title: 'CarNumberPlateDetection',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/CarNumberPlateDetection',
        media: [{ 'type': 'image', 'src': 'images/project/carnumberplatedetection.png', 'alt': 'CarNumberPlateDetection Interface' }],
        description: `Vehicle Number Plate Detection System! 🚗 This project leverages the powerful YOLOv8 framework to accurately detect vehicle number plates in images.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Vehicle Number Plate Detection System! 🚗 This project leverages the powerful YOLOv8 framework to accurately detect vehicle number plates in images.</p>

        <h4>Features</h4>
        <p>High Accuracy: Achieves a Mean Average Precision (mAP) of 99.2%.
Real-Time Detection: Optimized for fast inference suitable for real-time applications.
User-Friendly Interface: Built with Streamlit for easy interaction and visualization.
Scalable Training Pipeline: Utilize Google Colab for efficient model training with GPU support.
Comprehensive Documentation: Detailed README and usage guides to help you get started quickly.
Robust Error Handling: Informative messages to guide users through any issues.</p>

        <h4>Development Stack & Skills</h4>
        <p>python , YOLO, streamlit</p>
    `
    },

    'webscraperforimage': {
        title: 'WebScraperForImage',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/WebScraperForImage',
        media: [{ 'type': 'image', 'src': 'images/project/webscraperforimage.png', 'alt': 'WebScraperForImage Interface' }],
        description: `This project provides a FastAPI application that allows you to upload an image. The service uses Google Cloud Vision to perform reverse image search and extract text, and Selenium with headless Chrome to capture PDF screenshots of webpages where the image is found.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This project provides a FastAPI application that allows you to upload an image. The service uses Google Cloud Vision to perform reverse image search and extract text, and Selenium with headless Chrome to capture PDF screenshots of webpages where the image is found.</p>

        <h4>Features</h4>
        <p>Image Upload 📤: Easily upload an image using the provided web form.
Text Extraction 🔍: Extract text from the image using the Google Cloud Vision API.
Reverse Image Search 🕵️‍♂️: Identify webpages that include the uploaded image.
PDF Screenshot 📄: Generate PDF screenshots of each matching webpage.
CSV Logging 📝: Log results (timestamp, URL, and PDF filename) into a CSV file.
Interactive API Documentation 📑: Test the endpoint via Swagger UI at /docs.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, api, </p>
    `
    },

    'braintumorprediction': {
        title: 'BrainTumorPrediction',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/BrainTumorPrediction',
        media: [{ 'type': 'image', 'src': 'images/project/braintumorprediction.png', 'alt': 'BrainTumorPrediction Interface' }],
        description: `This tutorial will guide you through setting up the MRI 🩺 Image Classification app locally. The app allows users to upload MRI images, predict classifications, and visualize results, including tumor contouring. 🎨`,
        fullDescription: `
        <h4>Description</h4>
        <p>This tutorial will guide you through setting up the MRI 🩺 Image Classification app locally. The app allows users to upload MRI images, predict classifications, and visualize results, including tumor contouring. 🎨</p>

        <h4>Features</h4>
        <p>Upload MRI Image: Allows users to upload an MRI image for classification.
Prediction: Classifies MRI images into four categories.
 Contour Detection: Displays the original and contoured images side by side.
 Graph: Display the confidence graph.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit, api, </p>
    `
    },

    'handwrittendigitalsclassification': {
        title: 'HandWrittenDigitalsClassification',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/HandWrittenDigitalsClassification',
        media: [{ 'type': 'image', 'src': 'images/project/handwrittendigitalsclassification.png', 'alt': 'HandWrittenDigitalsClassification Interface' }],
        description: `Welcome to the MNIST Digit Recognition project! This repository demonstrates a complete workflow for building, training, and deploying a neural network to recognize handwritten digits from the MNIST dataset.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Welcome to the MNIST Digit Recognition project! This repository demonstrates a complete workflow for building, training, and deploying a neural network to recognize handwritten digits from the MNIST dataset.</p>

        <h4>Features</h4>
        <p>Model Training: Train a neural network to classify handwritten digits with high accuracy.
Interactive Drawing: Draw digits directly on a canvas for real-time classification.
 Image Upload: Upload digit images for prediction.
 Model Reusability: Save the trained model and reload it for predictions.
 Model Performance Metrics: Display detailed training and evaluation results.</p>

        <h4>Development Stack & Skills</h4>
        <p>python</p>
    `
    },

    'biomedicalliteraturehelperwithbiogpt': {
        title: 'BiomedicalLiteratureHelperWithBioGPT',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/BiomedicalLiteratureHelperWithBioGPT',
        media: [{ 'type': 'image', 'src': 'images/project/biomedicalliteraturehelperwithbiogpt.png', 'alt': 'BiomedicalLiteratureHelperWithBioGPT Interface' }],
        description: `Welcome to the Biomedical Literature Helper with BioGPT! This Streamlit application leverages the power of the BioGPT language model to assist researchers, clinicians, and enthusiasts in generating and mining biomedical texts efficiently.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Welcome to the Biomedical Literature Helper with BioGPT! This Streamlit application leverages the power of the BioGPT language model to assist researchers, clinicians, and enthusiasts in generating and mining biomedical texts efficiently.</p>

        <h4>Features</h4>
        <p>Advanced Text Generation: Utilize BioGPT to generate coherent and contextually relevant biomedical text based on your input prompts.
Multiple Model Options: Choose from various pre-trained BioGPT models to suit your specific needs.
Customizable Output: Adjust parameters like minimum and maximum sequence lengths, and the number of generated sequences.
User-Friendly Interface: Intuitive layout with real-time feedback and interactive elements.
Example Prompts: Explore pre-defined examples to understand the capabilities of BioGPT.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, GPT</p>
    `
    },

    'langgraphagent': {
        title: 'LangGraphAgent',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/LangGraphAgent',
        media: [{ 'type': 'image', 'src': 'images/project/langgraphagent.png', 'alt': 'LangGraphAgent Interface' }],
        description: `This repository presents an intelligent customer support agent powered by LangGraph—a robust tool designed for building intricate workflows with language models.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This repository presents an intelligent customer support agent powered by LangGraph—a robust tool designed for building intricate workflows with language models.</p>

        <h4>Features</h4>
        <p>State Management: Utilizes TypedDict to track and manage the state of each customer interaction.
Query Categorization: Classifies queries into categories such as Technical, Billing, or General.
Sentiment Analysis: Assesses the emotional tone of customer queries.
Response Generation: Constructs responses based on query category and sentiment.
Escalation Protocol: Automatically escalates queries with negative sentiment to a human agent.
Browser-Based Assistance: Employs the browser-use library to consult online sources for informed, up-to-date responses.
Workflow Graph: Leverages LangGraph to build an adaptable and extendable workflow.</p>

        <h4>Development Stack & Skills</h4>
        <p>python , streamlit</p>
    `
    },

    'finbert_ml_tradingbot': {
        title: 'FinBERT_ML_TradingBot',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/FinBERT_ML_TradingBot',
        media: [{ 'type': 'image', 'src': 'images/project/finbert_ml_tradingbot.png', 'alt': 'FinBERT_ML_TradingBot Interface' }],
        description: `ML Algorithmic Trading Bot (Python, Alpaca, FinBERT`,
        fullDescription: `
        <h4>Description</h4>
        <p>ML Algorithmic Trading Bot (Python, Alpaca, FinBERT</p>

        <h4>Features</h4>
        <p>Alpaca Integration: Uses the Alpaca API to place buy/sell orders and manage positions.
Sentiment Analysis: Leverages the FinBERT model (a pre-trained BERT model) to predict the sentiment of financial news headlines.
Backtesting: Includes the ability to backtest the strategy with historical data using Yahoo Finance.
Risk Management: Implements stop-loss and take-profit mechanisms for automated trade execution.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, api </p>
    `
    },

    'text_to_image_generation_via_dall_e3_stablediffusion': {
        title: 'Text-to-Image-Generation-via-DALL-E3-StableDiffusion',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/Text-to-Image-Generation-via-DALL-E3-StableDiffusion',
        media: [{ 'type': 'image', 'src': 'images/project/text_to_image_generation_via_dall_e3_stablediffusion.png', 'alt': 'Text-to-Image-Generation-via-DALL-E3-StableDiffusion Interface' }],
        description: `Streamlit-based web application that allows you to generate images from text prompts using cutting-edge AI models.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Streamlit-based web application that allows you to generate images from text prompts using cutting-edge AI models.</p>

        <h4>Features</h4>
        <p>Model Selection: Easily switch between OpenAI DALL-E 3 and Stable Diffusion.
User-Friendly Interface: A clean and responsive design with intuitive input fields.
Dynamic API Key Entry: For OpenAI DALL-E 3 users, simply enter your API key in the sidebar.
Enhanced Voice Input for Prompts:
Record Your Voice: Use the integrated voice recording feature to capture your spoken prompt.
Accurate Transcription with Whisper: The recorded audio is converted to the required format and transcribed using OpenAI's open-source Whisper model.
Automatic Prompt Update: The transcribed text is automatically inserted into the prompt field so you can easily review and edit it.
Image Generation: Generates a single image from your prompt and displays it directly in the browser.
NSFW Safety Bypass for Stable Diffusion: (⚠️ Note: The safety checker is disabled for Stable Diffusion to prevent blank (black) images. Use responsibly.)</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit, api, </p>
    `
    },

    'aiavatar': {
        title: 'AIAvatar',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/AIAvatar',
        media: [{ 'type': 'image', 'src': 'images/project/aiavatar.png', 'alt': 'AIAvatar Interface' }],
        description: `Welcome to the LightX AI Services app! This Streamlit application lets you transform your personal photo into either a personalized AI Avatar or a professional full-body cartoon character using the LightX APIs.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Welcome to the LightX AI Services app! This Streamlit application lets you transform your personal photo into either a personalized AI Avatar or a professional full-body cartoon character using the LightX APIs.</p>

        <h4>Features</h4>
        <p>Two Service Options:
AI Avatar: Generate a realistic, personalized avatar.
AI Cartoon: Transform your photo into a full-body cartoon character.
Simple Image Upload: Easy-to-use interface to upload your photo.
Custom Text Prompt: Provide a prompt to guide the style of the output.
Automated Polling: Automatically checks for order completion.
User-Friendly UI: Built with Streamlit for an interactive experience.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit, api, </p>
    `
    },

    'alpha_navigator': {
        title: 'Alpha-navigator',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/Alpha-navigator',
        media: [{ 'type': 'image', 'src': 'images/project/alpha_navigator.png', 'alt': 'Alpha-navigator Interface' }],
        description: `Alpha Navigator repository! This project demonstrates how to build a specialized conversational travel agent using LangChain, OpenAI GPT, and the Tavily search tool.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Alpha Navigator repository! This project demonstrates how to build a specialized conversational travel agent using LangChain, OpenAI GPT, and the Tavily search tool.</p>

        <h4>Features</h4>
        <p>OpenAI GPT Integration: Powered by a cutting-edge GPT model tailored for travel queries.
Live Tavily Search: Seamlessly integrates Tavily for up-to-date travel information.
Conversation Memory: Retains context using MemorySaver for multi-turn conversations.
Enhanced Streamlit UI: A modern, responsive interface with custom styling and a dedicated sidebar for configuration.
Travel-Specific Behavior: Only responds to travel-related queries; for non-travel questions, it replies with "I don't know."</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit, api, </p>
    `
    },

    'videoblur': {
        title: 'VideoBlur',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/VideoBlur',
        media: [{ 'type': 'image', 'src': 'images/project/videoblur.png', 'alt': 'VideoBlur Interface' }],
        description: `A simple yet effective solution to blur selected objects (like faces, people, etc.) in a video using YOLOv8.`,
        fullDescription: `
        <h4>Description</h4>
        <p>A simple yet effective solution to blur selected objects (like faces, people, etc.) in a video using YOLOv8.</p>

        <h4>Features</h4>
        <p>
Fast Object Detection: Leverages YOLOv8 for quick and reliable detection on video frames.
Configurable Blur Targets: Specify which classes (e.g., persons, cars, faces) to blur.
Easy to Integrate: Simple function calls to process each frame or entire videos.
High Accuracy: YOLOv8’s improved architecture ensures better precision and recall.</p>

        <h4>Development Stack & Skills</h4>
        <p>python</p>
    `
    },

    'medicaldiseasepredictionrecommendation': {
        title: 'MedicalDiseasePredictionRecommendation',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/MedicalDiseasePredictionRecommendation',
        media: [{ 'type': 'image', 'src': 'images/project/medicaldiseasepredictionrecommendation.png', 'alt': 'MedicalDiseasePredictionRecommendation Interface' }],
        description: `Medical Disease Prediction & Recommendation System! This project leverages machine learning to predict diseases based on user-input symptoms and then provides recommendations for precautions, medications, diet, and workouts.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Medical Disease Prediction & Recommendation System! This project leverages machine learning to predict diseases based on user-input symptoms and then provides recommendations for precautions, medications, diet, and workouts.</p>

        <h4>Features</h4>
        <p>Interactive User Interface: Designed with Streamlit and enhanced with custom CSS and a background image.
Disease Prediction: Uses a pre-trained SVM model to predict diseases based on selected symptoms.
Personalized Recommendations: Offers detailed advice on precautions, medications, diet, and workouts.
Dataset-Driven Insights: Leverages a comprehensive dataset from Kaggle.
Aesthetic Design: The UI features a visually appealing background image (embedded via base64 encoding) and modern styling.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, api, streamlit,</p>
    `
    },

    'blogcraft': {
        title: 'BlogCraft',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/BlogCraft',
        media: [{ 'type': 'image', 'src': 'images/project/blogcraft.png', 'alt': 'BlogCraft Interface' }],
        description: `BlogCraft is an AI-powered blog generation tool that helps you create detailed, engaging, and well-structured blog posts with ease.`,
        fullDescription: `
        <h4>Description</h4>
        <p>BlogCraft is an AI-powered blog generation tool that helps you create detailed, engaging, and well-structured blog posts with ease.</p>

        <h4>Features</h4>
        <p>Dual Mode Generation:
Choose between using OpenAI's GPT-3.5 (or GPT-4 if you have access) or the open source GPT-2 model for generating your blog posts.

User-Friendly Interface:
A modern, aesthetic UI with custom CSS for a clean and intuitive experience.

Easy to Configure:
Set up with minimal configuration using environment variables and a simple .env file.

Flexible Prompts:
Simply provide a blog topic and relevant interests; let the AI craft a comprehensive blog post that naturally incorporates your inputs.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, api, streamlit</p>
    `
    },

    'rag': {
        title: 'RAG',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/RAG',
        media: [{ 'type': 'image', 'src': 'images/project/rag.png', 'alt': 'RAG Interface' }],
        description: `This solution implements a Retrieval-Augmented Generation (RAG) technique, coupled with a Streamlit-based interface to manage PDF document uploads and queries.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This solution implements a Retrieval-Augmented Generation (RAG) technique, coupled with a Streamlit-based interface to manage PDF document uploads and queries.</p>

        <h4>Features</h4>
        <p>Streamlit Interface for smooth PDF uploads and user engagement.
Google Gemini API integration, utilizing the Gemini Pro model for top-tier language generation.
LlamaIndex support for rapid and scalable content indexing and retrieval.
Modular Architecture for straightforward maintenance and adaptability.
High-Efficiency QA processes that harness the power of your uploaded documents.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, api, streamlit</p>
    `
    },

    'multiply_v2': {
        title: 'multiPly_v2',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/multiPly_v2',
        media: [{ 'type': 'image', 'src': 'images/project/multiply_v2.png', 'alt': 'multiPly_v2 Interface' }],
        description: `MultiPly: Reconstruction of Multiple People from Monocular Video in the Wild (CVPR2024 Oral)`,
        fullDescription: `
        <h4>Description</h4>
        <p>MultiPly: Reconstruction of Multiple People from Monocular Video in the Wild (CVPR2024 Oral)</p>

        <h4>Features</h4>
        <p>nan</p>

        <h4>Development Stack & Skills</h4>
        <p>nan</p>
    `
    },

    'stockmarketpriceprediction': {
        title: 'StockMarketPricePrediction',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/StockMarketPricePrediction',
        media: [{ 'type': 'image', 'src': 'images/project/stockmarketpriceprediction.png', 'alt': 'StockMarketPricePrediction Interface' }],
        description: `Stock Price Prediction App, an interactive web application designed to forecast future stock prices using advanced time-series modeling techniques.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Stock Price Prediction App, an interactive web application designed to forecast future stock prices using advanced time-series modeling techniques.</p>

        <h4>Features</h4>
        <p>Single-Ticker Forecasting: Enter a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT) to retrieve and analyze historical data.
Interactive Plots: Visualize historical Open/Close prices using dynamic and responsive charts.
Prophet-Based Forecasting: Use a robust forecasting library that automatically handles trends and seasonality.
Customizable Horizon: Enter how many years you want to forecast, offering flexibility for short-term or mid-term predictions.
Forecast Components: View trend, weekly, and yearly seasonal insights (when relevant) in a user-friendly format.
Evaluation Metrics: (Optional in certain builds) Access mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), mean absolute percentage error (MAPE), and an approximate accuracy percentage for performance analysis.
Multi-Index Handling: Automatically flattens multi-level data returned by Yahoo Finance for seamless processing.
</p>

        <h4>Development Stack & Skills</h4>
        <p>python, streamlit , StockAPI</p>
    `
    },

    'whisper_android': {
        title: 'whisper_android',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/whisper_android',
        media: [{ 'type': 'image', 'src': 'images/project/whisper_android.png', 'alt': 'whisper_android Interface' }],
        description: `Offline Speech Recognition with OpenAI Whisper and TensorFlow Lite for Android`,
        fullDescription: `
        <h4>Description</h4>
        <p>Offline Speech Recognition with OpenAI Whisper and TensorFlow Lite for Android</p>

        <h4>Features</h4>
        <p>This repository offers two Android apps leveraging the OpenAI Whisper speech-to-text model. One app uses the TensorFlow Lite Java API for easy Java integration, while the other employs the TensorFlow Lite Native API for enhanced performance. It also includes a Python script for model generation and pre-built APKs for straightforward deployment.</p>

        <h4>Development Stack & Skills</h4>
        <p>C++
52.6%
 
C
29.8%
 
Python
3.6%
 
Java
2.2%
 
Starlark
1.8%
 
Rust
1.5%
</p>
    `
    },

    'videoaddsplatform': {
        title: 'VideoAddsPlatform',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/VideoAddsPlatform',
        media: [{ 'type': 'image', 'src': 'images/project/videoaddsplatform.png', 'alt': 'VideoAddsPlatform Interface' }],
        description: `A FastAPI-based platform that integrates YouTube and Google Ads APIs to search for videos and video advertisements.`,
        fullDescription: `
        <h4>Description</h4>
        <p>A FastAPI-based platform that integrates YouTube and Google Ads APIs to search for videos and video advertisements.</p>

        <h4>Features</h4>
        <p>Tubesift, a video ad research tool. The platform will allow users to search for YouTube videos and ads based on keywords, analyze their performance, and organize them into a library. Plus, setting up a monitoring system e.g. whenever a new ad pops up, for any monitored channel, there will be a notification by email. The system should retrieve and display all publicly available videos from a specific YouTube channel, including relevant metadata such as title, description, views, and engagement metrics.

Key Features:

Search for videos by keyword, channel, or category.
Filter results by video metrics (e.g., views, likes, comments).
Display analytics for videos (e.g., engagement rates, audience demographics).
Save videos into a user’s personal library for reference.</p>

        <h4>Development Stack & Skills</h4>
        <p>Python, Youtube API, fastAPI</p>
    `
    },

    'faseehgpt': {
        title: 'FaseehGPT',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/FaseehGPT',
        media: [{ 'type': 'image', 'src': 'images/project/faseehgpt.png', 'alt': 'FaseehGPT Interface' }],
        description: `FaseehGPT is an advanced pipeline for training a GPT-style language model specifically designed for the Arabic language.`,
        fullDescription: `
        <h4>Description</h4>
        <p>FaseehGPT is an advanced pipeline for training a GPT-style language model specifically designed for the Arabic language.</p>

        <h4>Features</h4>
        <p>Pre-trained Tokenizer: Utilizes asafaya/bert-base-arabic for robust Arabic tokenization.
Efficient Data Preprocessing: Handles large datasets and long sequences with overlapping chunking.
Customizable Model Architecture: Allows for easy adjustments in model dimensions, attention heads, and layers.
Training Pipeline: Fully equipped for model training, checkpoint saving, and evaluation.
Text Generation: Generates high-quality Arabic text from a trained model.
Evaluation Metrics: Supports perplexity and BLEU score for model performance evaluation.</p>

        <h4>Development Stack & Skills</h4>
        <p>python, LLM, GPT</p>
    `
    },

    'project_proposal_agent': {
        title: 'project-proposal-agent',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/project-proposal-agent',
        media: [{ 'type': 'image', 'src': 'images/project/project_proposal_agent.png', 'alt': 'project-proposal-agent Interface' }],
        description: `An intelligent system that processes business requirement documents and generates comprehensive technical documentation using AI-powered analysis.`,
        fullDescription: `
        <h4>Description</h4>
        <p>An intelligent system that processes business requirement documents and generates comprehensive technical documentation using AI-powered analysis.</p>

        <h4>Features</h4>
        <p>Document processing support for multiple formats (PDF, DOCX, TXT)
AI-powered analysis using Groq LLM:
Requirements analysis
Technical specifications generation
Architecture recommendations
Automated project planning and task breakdown
Detailed cost estimation including:
Labor costs with complexity factors
Infrastructure costs
License costs
Interactive web interface using Streamlit
JSON-formatted outputs for all analyses
Downloadable consolidated reports</p>

        <h4>Development Stack & Skills</h4>
        <p>Python, Groq</p>
    `
    },

    'detectron2_customized': {
        title: 'detectron2-customized',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/detectron2-customized',
        media: [{ 'type': 'image', 'src': 'images/project/detectron2_customized.png', 'alt': 'detectron2-customized Interface' }],
        description: `Detectron2 is a platform for object detection, segmentation and other visual recognition tasks.`,
        fullDescription: `
        <h4>Description</h4>
        <p>Detectron2 is a platform for object detection, segmentation and other visual recognition tasks.</p>

        <h4>Features</h4>
        <p>Includes new capabilities such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend, DeepLab, ViTDet, MViTv2 etc.
Used as a library to support building research projects on top of it.
Models can be exported to TorchScript format or Caffe2 format for deployment.
It trains much faster.</p>

        <h4>Development Stack & Skills</h4>
        <p>Python, Cuda, C++</p>
    `
    },

    'texture_transfer': {
        title: 'Texture Transfer Onto Torso for Soccer Players',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'September 2025',
        url: 'https://github.com/alphatechlogics/detectron2-customized',
        media: [{ 'type': 'image', 'src': 'images/project/detectron2_customized.png', 'alt': 'detectron2-customized Interface' }],
        description: `This project implements automated logo placement on football player jerseys with consistent tracking across video frames. After testing 18 state-of-the-art models, we developed two primary solutions: CameraHMR for 3D mesh-based texture transfer and DensePose+SAPIENS+YOLO pipeline for real-time processing.`,
        fullDescription: `
        <h4>Description</h4>
        <p>This project implements automated logo placement on football player jerseys with consistent tracking across video frames. After testing 18 state-of-the-art models, we developed two primary solutions: CameraHMR for 3D mesh-based texture transfer and DensePose+SAPIENS+YOLO pipeline for real-time processing.</p>

        <h4>Features</h4>
        <p>Real-time logo placement with 3D mesh-based texture mapping, temporal consistency through frame smoothing and UV coordinate stabilization, multi-player processing with instance-level segmentation, optimized performance achieving 3-4x speed improvement, and live webcam support for real-time applications.</p>

        <h4>Development Stack & Skills</h4>
        <p>Computer Vision: OpenCV, DensePose, CameraHMR, SAPIENS, YOLO
Deep Learning: PyTorch, 3D Human Pose Estimation, Instance Segmentation
3D Processing: Mesh reconstruction, UV mapping, Texture sampling and blending
Optimization: Vectorized operations, Memory management, Real-time processing</p>
    `
    },


    'multiply_customization': {
        title: 'MultiPly-customization',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/alphatechlogics/MultiPly-customization',
        media: [{ 'type': 'image', 'src': 'images/project/multiply_customization.png', 'alt': 'MultiPly-customization Interface' }],
        description: `
MultiPly: Reconstruction of Multiple People from Monocular Video in the Wild (CVPR2024 Oral)`,
        fullDescription: `
        <h4>Description</h4>
        <p>
MultiPly: Reconstruction of Multiple People from Monocular Video in the Wild (CVPR2024 Oral)</p>

        <h4>Features</h4>
        <p>Reconstruction of Multiple People from Monocular Video in the Wild</p>

        <h4>Development Stack & Skills</h4>
        <p>Python, Cuda</p>
    `
    }
};

// Function to get URL parameters
function getUrlParameter(name) {
    name = name.replace(/[\[]/, '\\[').replace(/[\]]/, '\\]');
    const regex = new RegExp('[\\?&]' + name + '=([^&#]*)');
    const results = regex.exec(location.search);
    return results === null ? '' : decodeURIComponent(results[1].replace(/\+/g, ' '));
}

// Function to load project details
function loadProjectDetails() {
    const projectId = getUrlParameter('project');
    const project = projectData[projectId];

    if (!project) {
        // If no project ID or invalid ID, show default content
        document.querySelector('.page-title h1').textContent = 'Project Details';
        document.querySelector('.page-title p').textContent = 'Project not found. Please select a valid project.';
        return;
    }

    // Update page title and description
    document.querySelector('.page-title h1').textContent = project.title;
    document.querySelector('.page-title p').textContent = project.description;

    // Update document title
    document.title = `${project.title} - ATL`;

    // Update project media in swiper
    const swiperWrapper = document.querySelector('.swiper-wrapper');
    swiperWrapper.innerHTML = '';

    project.media.forEach((media, index) => {
        const slide = document.createElement('div');
        slide.className = 'swiper-slide';

        if (media.type === 'video') {
            slide.innerHTML = `
                <video 
                    controls 
                    poster="${media.poster}" 
                    style="width: 100%; height: auto; max-height: 500px; object-fit: cover;"
                >
                    <source src="${media.src}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
        } else {
            slide.innerHTML = `<img src="${media.src}" alt="${media.alt || project.title + ' - Image ' + (index + 1)}">`;
        }

        swiperWrapper.appendChild(slide);
    });

    // Update project information
    const portfolioInfo = document.querySelector('.portfolio-info ul');
    portfolioInfo.innerHTML = `
        <li><strong>Category</strong>: ${project.category}</li>
        <li><strong>Client</strong>: ${project.client}</li>
        <li><strong>Project date</strong>: ${project.date}</li>
        ${project.url ? `<li><strong>Project URL</strong>: <a href="${project.url}" target="_blank">View Project</a></li>` : ''}
    `;

    // Update description
    const portfolioDescription = document.querySelector('.portfolio-description');
    portfolioDescription.innerHTML = `
        <h2>${project.title}</h2>
        ${project.fullDescription}
    `;

    // Reinitialize Swiper if it exists
    if (typeof Swiper !== 'undefined') {
        setTimeout(() => {
            new Swiper('.portfolio-details-slider', {
                loop: true,
                speed: 600,
                autoplay: {
                    delay: 5000
                },
                slidesPerView: 'auto',
                pagination: {
                    el: '.swiper-pagination',
                    type: 'bullets',
                    clickable: true
                }
            });
        }, 100);
    }
}

// Load project details when the page loads
document.addEventListener('DOMContentLoaded', loadProjectDetails);