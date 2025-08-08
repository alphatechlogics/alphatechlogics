// Project data
const projectData = {
    'news_summarizer': {
        title: 'AI News Summarizer',
        client: 'Zigoland',
        category: 'Artificial Intelligence',
        date: 'March 2024',
        url: 'https://github.com/your-repo/news-summarizer',
        media: [
            { type: 'image', src: 'images/project/news_summarizer.png', alt: 'News Summarizer Interface' },
            { type: 'video', src: 'assets/videos/news_summarizer_demo.mp4', poster: 'assets/img/portfolio/news_summarizer_2.png' },
            { type: 'image', src: 'assets/img/portfolio/news_summarizer_3.png', alt: 'News Summarizer Dashboard' }
        ],
        description: 'An advanced AI-powered news summarization system that processes multiple news sources and generates concise, accurate summaries using natural language processing. The system aggregates news from various sources, analyzes content for relevance and importance, and produces digestible summaries while maintaining the core message and context.',
        fullDescription: `
            <h4>Project Overview</h4>
            <p>The AI News Summarizer represents a cutting-edge solution for information consumption in our fast-paced digital world. Built using state-of-the-art natural language processing techniques, this system revolutionizes how users interact with news content.</p>
            
            <h4>Key Features</h4>
            <ul>
                <li><strong>Multi-source aggregation:</strong> Collects news from 50+ reliable sources</li>
                <li><strong>Real-time processing:</strong> Updates summaries every 15 minutes</li>
                <li><strong>Sentiment analysis:</strong> Provides emotional context to news stories</li>
                <li><strong>Topic clustering:</strong> Groups related stories for comprehensive coverage</li>
                <li><strong>Bias detection:</strong> Identifies and flags potentially biased content</li>
            </ul>
            
            <h4>Technical Implementation</h4>
            <p>The system utilizes advanced transformer models including BERT and GPT for text understanding and generation. The backend is built with Python and Flask, while the frontend uses React for a responsive user interface.</p>
            
            <h4>Impact & Results</h4>
            <p>Successfully reduced news consumption time by 75% while maintaining 95% information retention among test users. The system processes over 10,000 articles daily and serves summaries to 50,000+ active users.</p>
        `
    },
    'multimodel_rag': {
        title: 'Multimodal RAG System',
        client: 'Visodream',
        category: 'Machine Learning',
        date: 'February 2024',
        url: 'https://github.com/your-repo/multimodal-rag',
        media: [
            { type: 'image', src: 'assets/img/portfolio/multimodel_rag.png', alt: 'Multimodal RAG Architecture' },
            { type: 'video', src: 'assets/videos/multimodel_rag_demo.mp4', poster: 'assets/img/portfolio/multimodel_rag_poster.png' },
            { type: 'image', src: 'assets/img/portfolio/multimodel_rag_3.png', alt: 'RAG System Interface' }
        ],
        description: 'A sophisticated Retrieval-Augmented Generation system that processes both text and visual content to provide comprehensive responses. This system combines the power of large language models with advanced retrieval mechanisms to deliver contextually accurate and relevant information.',
        fullDescription: `
            <h4>Revolutionary Approach</h4>
            <p>Our Multimodal RAG system breaks traditional boundaries by seamlessly integrating text and visual information processing. This creates a more comprehensive understanding of user queries and enables richer, more accurate responses.</p>
            
            <h4>Core Capabilities</h4>
            <ul>
                <li><strong>Dual-mode processing:</strong> Simultaneously handles text and image inputs</li>
                <li><strong>Semantic search:</strong> Uses vector embeddings for precise information retrieval</li>
                <li><strong>Context preservation:</strong> Maintains conversation history for coherent responses</li>
                <li><strong>Multi-document synthesis:</strong> Combines information from multiple sources</li>
                <li><strong>Real-time adaptation:</strong> Learns from user interactions to improve accuracy</li>
            </ul>
            
            <h4>Architecture</h4>
            <p>Built on a microservices architecture using FastAPI, with separate services for text processing, image analysis, vector storage, and response generation. Uses OpenAI's GPT models combined with CLIP for multimodal understanding.</p>
            
            <h4>Business Impact</h4>
            <p>Improved query response accuracy by 40% compared to text-only systems. Reduced customer support workload by 60% through automated, intelligent responses.</p>
        `
    },
    'sentiment_analysis': {
        title: 'Social Media Sentiment Analysis',
        client: 'Primoday',
        category: 'Data Analytics',
        date: 'January 2024',
        url: 'https://github.com/your-repo/sentiment-analysis',
        media: [
            { type: 'image', src: 'assets/img/portfolio/sentimental_analysis.jpg', alt: 'Sentiment Analysis Dashboard' },
            { type: 'image', src: 'assets/img/portfolio/sentiment_dashboard.png', alt: 'Real-time Dashboard' },
            { type: 'video', src: 'assets/videos/sentiment_analysis_demo.mp4', poster: 'assets/img/portfolio/sentiment_trends.png' }
        ],
        description: 'A comprehensive sentiment analysis platform that monitors social media mentions and provides real-time insights into brand perception. The system tracks sentiment across multiple platforms and provides actionable insights for brand management.',
        fullDescription: `
            <h4>Comprehensive Social Monitoring</h4>
            <p>Our sentiment analysis platform provides unprecedented insights into brand perception across social media platforms. By analyzing millions of posts, comments, and mentions, we deliver real-time sentiment tracking and trend analysis.</p>
            
            <h4>Advanced Features</h4>
            <ul>
                <li><strong>Multi-platform monitoring:</strong> Tracks Twitter, Facebook, Instagram, LinkedIn, and Reddit</li>
                <li><strong>Real-time alerts:</strong> Instant notifications for sentiment spikes or brand mentions</li>
                <li><strong>Trend analysis:</strong> Historical sentiment patterns and forecasting</li>
                <li><strong>Competitor comparison:</strong> Benchmark sentiment against competitors</li>
                <li><strong>Crisis detection:</strong> Early warning system for potential PR issues</li>
            </ul>
            
            <h4>Technical Excellence</h4>
            <p>Utilizes advanced NLP techniques including BERT, VADER, and custom-trained models. The system processes 100,000+ posts daily with 85% accuracy in sentiment classification across 12 languages.</p>
            
            <h4>Business Results</h4>
            <p>Enabled proactive brand management, reducing negative sentiment impact by 45%. Improved customer satisfaction scores by 30% through timely response to concerns.</p>
        `
    },
    'math_tutor': {
        title: 'AI Math Tutor',
        client: 'Nextlite',
        category: 'Education Technology',
        date: 'December 2023',
        url: 'https://github.com/your-repo/ai-math-tutor',
        media: [
            { type: 'video', src: 'assets/videos/math_tutor_demo.mp4', poster: 'assets/img/portfolio/math_tutor.png' },
            { type: 'image', src: 'assets/img/portfolio/math_interface.png', alt: 'Math Tutor Interface' },
            { type: 'image', src: 'assets/img/portfolio/math_progress.png', alt: 'Student Progress Tracking' }
        ],
        description: 'An intelligent tutoring system that provides personalized math instruction and step-by-step problem solving guidance. The system adapts to individual learning styles and provides immediate feedback to enhance the learning experience.',
        fullDescription: `
            <h4>Personalized Learning Revolution</h4>
            <p>Our AI Math Tutor transforms traditional mathematics education by providing personalized, adaptive instruction that meets each student where they are in their learning journey.</p>
            
            <h4>Innovative Features</h4>
            <ul>
                <li><strong>Adaptive learning paths:</strong> Customizes difficulty based on student performance</li>
                <li><strong>Step-by-step guidance:</strong> Breaks down complex problems into manageable steps</li>
                <li><strong>Visual problem solving:</strong> Interactive graphs and diagrams for better understanding</li>
                <li><strong>Progress tracking:</strong> Detailed analytics for students, parents, and teachers</li>
                <li><strong>Gamification:</strong> Achievement badges and progress rewards for motivation</li>
            </ul>
            
            <h4>AI-Powered Intelligence</h4>
            <p>Leverages machine learning algorithms to understand student learning patterns and adapt instruction accordingly. Uses computer vision to analyze handwritten work and provide real-time feedback.</p>
            
            <h4>Educational Impact</h4>
            <p>Improved student test scores by 30% and engagement rates by 60%. Successfully deployed in 200+ schools, serving over 15,000 students daily.</p>
        `
    },
    'fraud_detection': {
        title: 'ML Fraud Detection System',
        client: 'Syncnow',
        category: 'Financial Technology',
        date: 'November 2023',
        url: 'https://github.com/your-repo/fraud-detection',
        media: [
            { type: 'image', src: 'assets/img/portfolio/fraud_detection.jpg', alt: 'Fraud Detection System' },
            { type: 'video', src: 'assets/videos/fraud_detection_demo.mp4', poster: 'assets/img/portfolio/fraud_dashboard.png' },
            { type: 'image', src: 'assets/img/portfolio/fraud_analytics.png', alt: 'Fraud Analytics Dashboard' }
        ],
        description: 'A machine learning-based fraud detection system that analyzes transaction patterns to identify suspicious activities in real-time. The system provides comprehensive risk assessment and automated response mechanisms.',
        fullDescription: `
            <h4>Advanced Fraud Prevention</h4>
            <p>Our ML-powered fraud detection system represents the next generation of financial security, combining multiple detection algorithms to identify and prevent fraudulent transactions with unprecedented accuracy.</p>
            
            <h4>Sophisticated Detection Methods</h4>
            <ul>
                <li><strong>Behavioral analysis:</strong> Learns normal user patterns to detect anomalies</li>
                <li><strong>Real-time scoring:</strong> Instant risk assessment for every transaction</li>
                <li><strong>Network analysis:</strong> Identifies fraud rings and related suspicious activities</li>
                <li><strong>Device fingerprinting:</strong> Tracks device characteristics for additional security</li>
                <li><strong>Adaptive learning:</strong> Continuously improves from new fraud patterns</li>
            </ul>
            
            <h4>Technical Architecture</h4>
            <p>Built using ensemble machine learning methods including Random Forest, XGBoost, and Neural Networks. Processes over 1 million transactions daily with sub-100ms response times using Apache Kafka for real-time streaming.</p>
            
            <h4>Security Results</h4>
            <p>Reduced fraudulent transactions by 78% while maintaining 99.2% legitimate transaction approval rate. Saved clients over $50M in potential fraud losses in the first year.</p>
        `
    },
    'plant_disease': {
        title: 'Plant Disease Detection',
        client: 'Shifter',
        category: 'Agricultural Technology',
        date: 'October 2023',
        url: 'https://github.com/your-repo/plant-disease-detection',
        media: [
            { type: 'image', src: 'assets/img/portfolio/plant_disease.jpg', alt: 'Plant Disease Detection' },
            { type: 'image', src: 'assets/img/portfolio/plant_analysis.png', alt: 'Disease Analysis Results' },
            { type: 'video', src: 'assets/videos/plant_disease_demo.mp4', poster: 'assets/img/portfolio/plant_mobile.png' }
        ],
        description: 'A computer vision system that identifies plant diseases from leaf images and provides treatment recommendations to farmers. The system works across multiple crop types and provides actionable insights for crop management.',
        fullDescription: `
            <h4>Agricultural Innovation</h4>
            <p>Our plant disease detection system empowers farmers with AI-driven diagnostic capabilities, transforming traditional agriculture through early disease detection and precision treatment recommendations.</p>
            
            <h4>Comprehensive Plant Health</h4>
            <ul>
                <li><strong>Multi-crop support:</strong> Covers 20+ major crop types including tomatoes, potatoes, and corn</li>
                <li><strong>Disease identification:</strong> Detects 50+ common plant diseases with 92% accuracy</li>
                <li><strong>Treatment recommendations:</strong> Provides specific, actionable treatment plans</li>
                <li><strong>Offline capability:</strong> Works without internet connectivity for remote areas</li>
                <li><strong>Progress tracking:</strong> Monitors treatment effectiveness over time</li>
            </ul>
            
            <h4>Advanced Computer Vision</h4>
            <p>Utilizes convolutional neural networks trained on 100,000+ plant images. The mobile app uses TensorFlow Lite for on-device inference, ensuring fast diagnosis even in areas with poor connectivity.</p>
            
            <h4>Agricultural Impact</h4>
            <p>Achieved 92% accuracy in disease identification, helping farmers reduce crop loss by 45%. Currently used by over 10,000 farmers across 15 countries, protecting millions of dollars worth of crops.</p>
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