import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import numpy as np
from datetime import datetime

# Download NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Page Configuration
st.set_page_config(
    page_title="Resume Job Match Scorer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State for animations
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'animation_progress' not in st.session_state:
    st.session_state.animation_progress = 0


# ============= HELPER FUNCTIONS =============

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""


def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    """Remove common stopwords from text"""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])


def calculate_similarity(resume_text, job_description):
    """Calculate similarity score between resume and job description"""
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed


def extract_keywords(text, num_keywords=15):
    """Extract top keywords/terms from text"""
    words = word_tokenize(text.lower())
    words = [w for w in words if len(w) > 2 and w.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    word_freq = Counter(words)
    return word_freq.most_common(num_keywords)


def get_missing_keywords(resume_keywords, job_keywords):
    """Get keywords from job description missing in resume"""
    resume_set = set([word for word, _ in resume_keywords])
    job_set = set([word for word, _ in job_keywords])
    missing = job_set - resume_set
    job_dict = dict(job_keywords)
    missing_with_freq = [(word, job_dict[word]) for word in missing]
    return sorted(missing_with_freq, key=lambda x: x[1], reverse=True)[:10]


def generate_improvement_suggestions(resume_text, job_description, similarity_score, missing_keywords):
    """Generate actionable improvement suggestions"""
    suggestions = []
    
    resume_length = len(resume_text.split())
    job_length = len(job_description.split())
    
    if resume_length < job_length * 0.3:
        suggestions.append({
            'emoji': '📝',
            'title': 'Expand Your Resume',
            'description': 'Your resume is quite brief. Add more details about achievements and responsibilities.',
            'priority': 'High'
        })
    
    if missing_keywords:
        top_missing = [word for word, _ in missing_keywords[:5]]
        suggestions.append({
            'emoji': '🎯',
            'title': 'Add Missing Keywords',
            'description': f'Incorporate: {", ".join(top_missing)}',
            'priority': 'High'
        })
    
    if similarity_score < 40:
        suggestions.append({
            'emoji': '🔄',
            'title': 'Major Revision Needed',
            'description': 'Significantly rewrite focusing on job requirements.',
            'priority': 'Critical'
        })
    elif similarity_score < 60:
        suggestions.append({
            'emoji': '✏️',
            'title': 'Moderate Updates Recommended',
            'description': 'Add specific examples and metrics related to job requirements.',
            'priority': 'High'
        })
    elif similarity_score < 70:
        suggestions.append({
            'emoji': '🎁',
            'title': 'Small Polish Needed',
            'description': 'Add a few more relevant keywords and achievements.',
            'priority': 'Medium'
        })
    else:
        suggestions.append({
            'emoji': '⭐',
            'title': 'Outstanding Match',
            'description': 'Your resume is excellently tailored for this role!',
            'priority': 'Low'
        })
    
    return suggestions


def create_aesthetic_performance_bar(score):
    """Create a beautiful animated performance bar with smooth aesthetics"""
    fig, ax = plt.subplots(figsize=(14, 3), facecolor='#f8f9fa')
    
    # Determine colors based on score
    if score >= 70:
        main_color = '#10b981'  # Green
        gradient_color = '#059669'
        status = 'Excellent Match'
    elif score >= 50:
        main_color = '#3b82f6'  # Blue
        gradient_color = '#1d4ed8'
        status = 'Good Match'
    elif score >= 30:
        main_color = '#f59e0b'  # Amber
        gradient_color = '#d97706'
        status = 'Needs Work'
    else:
        main_color = '#ef4444'  # Red
        gradient_color = '#dc2626'
        status = 'Low Match'
    
    # Background track (empty bar)
    bg_bar = FancyBboxPatch(
        (0.5, 1.5), 98, 2,
        boxstyle="round,pad=0.2",
        edgecolor='#e5e7eb',
        facecolor='#e5e7eb',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(bg_bar)
    
    # Filled bar with gradient effect
    filled_width = min(score, 100)
    filled_bar = FancyBboxPatch(
        (0.5, 1.5), filled_width, 2,
        boxstyle="round,pad=0.2",
        edgecolor=gradient_color,
        facecolor=main_color,
        linewidth=2.5,
        alpha=0.95
    )
    ax.add_patch(filled_bar)
    
    # Add glow effect
    glow_bar = FancyBboxPatch(
        (0.5, 1.5), filled_width, 2,
        boxstyle="round,pad=0.2",
        edgecolor=main_color,
        facecolor='none',
        linewidth=4,
        alpha=0.2
    )
    ax.add_patch(glow_bar)
    
    # Percentage text in the middle of the bar
    ax.text(
        filled_width / 2, 2.5,
        f'{score:.1f}%',
        fontsize=42,
        fontweight='bold',
        color='white',
        ha='center',
        va='center',
        family='sans-serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=main_color, edgecolor='none', alpha=0)
    )
    
    # Status label to the right
    ax.text(
        102, 2.5,
        status,
        fontsize=18,
        fontweight='600',
        color=main_color,
        ha='left',
        va='center',
        family='sans-serif'
    )
    
    # Milestone markers
    milestones = [25, 50, 75, 100]
    for milestone in milestones:
        ax.plot([milestone, milestone], [0.8, 1.5], color='#d1d5db', linewidth=1, alpha=0.5, linestyle='--')
        if milestone < 100:
            ax.text(milestone, 0.5, f'{milestone}%', fontsize=9, ha='center', color='#9ca3af', fontweight='500')
    
    # 0% and 100% labels
    ax.text(0.5, 0.5, '0%', fontsize=9, ha='left', color='#9ca3af', fontweight='500')
    ax.text(100, 0.5, '100%', fontsize=9, ha='right', color='#9ca3af', fontweight='500')
    
    # Set axis properties
    ax.set_xlim(-5, 115)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_detailed_performance_metrics(score):
    """Create detailed metrics visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), facecolor='#f8f9fa')
    
    # Determine colors based on score
    if score >= 70:
        main_color = '#10b981'
    elif score >= 50:
        main_color = '#3b82f6'
    elif score >= 30:
        main_color = '#f59e0b'
    else:
        main_color = '#ef4444'
    
    # ============ Score Gauge (Circular) ============
    # Create circular gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # Background circle
    ax1.plot(r * np.cos(theta), r * np.sin(theta), color='#e5e7eb', linewidth=20)
    
    # Filled circle
    theta_fill = np.linspace(0, np.pi * (score / 100), 100)
    ax1.plot(r * np.cos(theta_fill), r * np.sin(theta_fill), color=main_color, linewidth=20)
    
    # Center circle (to create the gauge effect)
    circle = Circle((0, 0), 0.6, color='#f8f9fa', zorder=10)
    ax1.add_patch(circle)
    
    # Add text in center
    ax1.text(0, 0, f'{score:.1f}%', fontsize=48, fontweight='bold', color=main_color, ha='center', va='center')
    
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-0.5, 1.3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Overall Match Score', fontsize=16, fontweight='bold', pad=20, color='#1e293b')
    
    # ============ Category Breakdown ============
    categories = ['Keywords\nMatch', 'Content\nAlignment', 'Structure\nQuality', 'Relevance\nScore']
    
    # Simulate category scores based on overall score
    if score >= 70:
        cat_scores = [85, 75, 70, 80]
    elif score >= 50:
        cat_scores = [65, 55, 50, 60]
    elif score >= 30:
        cat_scores = [45, 35, 40, 38]
    else:
        cat_scores = [25, 20, 22, 25]
    
    bars = ax2.barh(categories, cat_scores, color=[main_color if s >= 50 else '#fca5a5' for s in cat_scores], 
                     edgecolor='#d1d5db', linewidth=2, height=0.6, alpha=0.85)
    
    # Add value labels on bars
    for i, (bar, score_val) in enumerate(zip(bars, cat_scores)):
        ax2.text(score_val + 2, i, f'{score_val}%', va='center', fontweight='600', fontsize=11, color='#1e293b')
    
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Score (%)', fontsize=11, fontweight='600', color='#64748b')
    ax2.set_title('Category Breakdown', fontsize=16, fontweight='bold', pad=20, color='#1e293b')
    ax2.grid(axis='x', alpha=0.2, linestyle='--')
    ax2.set_axisbelow(True)
    
    # ============ Match Level Distribution ============
    levels = ['0-25%', '25-50%', '50-75%', '75-100%']
    colors_dist = ['#fca5a5', '#fcd34d', '#86efac', '#6ee7b7']
    
    if score < 25:
        sizes = [100, 0, 0, 0]
    elif score < 50:
        sizes = [100 - score, score, 0, 0]
    elif score < 75:
        sizes = [0, 100 - score, score - 50, 0]
    else:
        sizes = [0, 0, 100 - score, score - 75]
    
    wedges, texts, autotexts = ax3.pie(
        [25, 25, 25, 25],
        labels=levels,
        colors=colors_dist,
        autopct='%1.0f%%',
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': '600'},
        wedgeprops={'edgecolor': '#f8f9fa', 'linewidth': 2}
    )
    
    # Highlight current position
    current_angle = 90 - (score / 100) * 360
    ax3.plot([0], [0], 'o', markersize=15, color=main_color, zorder=10)
    
    ax3.set_title('Performance Level', fontsize=16, fontweight='bold', pad=20, color='#1e293b')
    
    # ============ Recommendation Score ============
    recommendations = ['Add Keywords', 'Expand Content', 'Improve Formatting', 'Enhance Skills']
    rec_scores = [100 - score, max(0, 70 - score), max(0, 60 - score), max(0, 80 - score)]
    
    colors_rec = ['#ef4444' if s > 40 else '#f59e0b' if s > 20 else '#10b981' for s in rec_scores]
    
    bars = ax4.bar(range(len(recommendations)), rec_scores, color=colors_rec, 
                   edgecolor='#d1d5db', linewidth=2, alpha=0.85)
    
    # Add value labels on top of bars
    for bar, rec_score in zip(bars, rec_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{rec_score:.0f}%',
                ha='center', va='bottom', fontweight='600', fontsize=11, color='#1e293b')
    
    ax4.set_xticks(range(len(recommendations)))
    ax4.set_xticklabels(recommendations, fontsize=10, fontweight='600')
    ax4.set_ylabel('Priority Level (%)', fontsize=11, fontweight='600', color='#64748b')
    ax4.set_ylim(0, 100)
    ax4.set_title('Action Items Priority', fontsize=16, fontweight='bold', pad=20, color='#1e293b')
    ax4.grid(axis='y', alpha=0.2, linestyle='--')
    ax4.set_axisbelow(True)
    
    plt.suptitle('Detailed Performance Analysis', fontsize=18, fontweight='bold', y=0.995, color='#1e293b')
    plt.tight_layout()
    
    return fig


# ============= MAIN APP =============

# Header
st.markdown("# 📄 Resume Job Match Scorer")
st.markdown("### Analyze how perfectly your resume aligns with job requirements")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About This Tool")
    st.info("""
    This intelligent tool helps you:
    - **Match Score**: See how well your resume fits the job
    - **Keyword Analysis**: Identify important job requirements
    - **Improvement Tips**: Get actionable suggestions to enhance your resume
    - **Smart Matching**: Uses TF-IDF + Cosine Similarity algorithm
    """)
    
    st.markdown("### 🚀 How It Works")
    st.write("""
    1. **Upload Resume** - PDF format
    2. **Enter Job Title** - What position are you applying for?
    3. **Paste Job Description** - Copy from the job posting
    4. **Analyze Match** - Click to see detailed results
    5. **Review Suggestions** - Get improvement recommendations
    """)
    
    st.markdown("### 💡 Pro Tips")
    st.write("""
    - Use complete job descriptions for better results
    - Tailor your resume to specific job keywords
    - Aim for 70%+ match score
    - Update your resume based on suggestions
    """)
    
    st.divider()
    st.caption("Made with ❤️ using Streamlit")


# Input Section
st.markdown("## 📥 Input Your Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], label_visibility="collapsed")

with col2:
    st.markdown("### 💼 Job Title")
    job_title = st.text_input(
        "Job title",
        placeholder="e.g., Senior Software Engineer",
        label_visibility="collapsed"
    )

st.markdown("### 📋 Job Description")
job_description = st.text_area(
    "Job description",
    height=200,
    placeholder="Paste the complete job description here...",
    label_visibility="collapsed"
)

# Analyze Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🚀 Analyze Match", use_container_width=True, type="primary")

# Results Section
if analyze_button:
    # Validation
    if not uploaded_file:
        st.error("❌ Please upload your resume (PDF)")
        st.stop()
    
    if not job_description:
        st.error("❌ Please paste the job description")
        st.stop()
    
    if not job_title:
        st.warning("⚠️ Job title helps provide better suggestions!")
    
    # Analysis with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Extract resume
    status_text.text("📖 Extracting resume text...")
    progress_bar.progress(20)
    time.sleep(0.3)
    
    resume_text = extract_text_from_pdf(uploaded_file)
    if not resume_text:
        st.error("❌ Could not extract text from PDF. Please try another file.")
        st.stop()
    
    # Calculate similarity
    status_text.text("🔍 Calculating match score...")
    progress_bar.progress(40)
    time.sleep(0.3)
    
    similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)
    
    # Extract keywords
    status_text.text("🏷️ Analyzing keywords...")
    progress_bar.progress(60)
    time.sleep(0.3)
    
    resume_keywords = extract_keywords(resume_processed, 15)
    job_keywords = extract_keywords(job_processed, 15)
    missing_keywords = get_missing_keywords(resume_keywords, job_keywords)
    
    # Generate suggestions
    status_text.text("💡 Generating suggestions...")
    progress_bar.progress(80)
    time.sleep(0.3)
    
    suggestions = generate_improvement_suggestions(
        resume_text, job_description, similarity_score, missing_keywords
    )
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Results
    st.divider()
    st.markdown("## 📊 Your Match Analysis")
    
    # Score Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric("Match Score", f"{similarity_score:.1f}%")
    
    with col2:
        st.metric("Job Title", job_title if job_title else "Not provided")
    
    with col3:
        resume_words = len(resume_text.split())
        st.metric("Resume Length", f"{resume_words} words")
    
    st.divider()
    
    # Beautiful Performance Bar
    st.markdown("### ✨ Performance Visualization")
    fig = create_aesthetic_performance_bar(similarity_score)
    st.pyplot(fig, use_container_width=True)
    
    st.divider()
    
    # Status Message
    if similarity_score >= 70:
        st.success("""
        ✨ **Excellent Match!** 
        Your resume strongly aligns with the job requirements. You're well-positioned for this role!
        """)
    elif similarity_score >= 50:
        st.info("""
        👍 **Good Match!** 
        Your resume aligns fairly well with the job description. Consider the suggestions below to strengthen it further.
        """)
    else:
        st.warning("""
        ⚠️ **Low Match!** 
        Your resume needs significant adjustments. Follow the improvement suggestions to better align with the job requirements.
        """)
    
    st.divider()
    
    # Detailed Performance Metrics
    st.markdown("### 📈 Detailed Performance Metrics")
    fig_metrics = create_detailed_performance_metrics(similarity_score)
    st.pyplot(fig_metrics, use_container_width=True)
    
    st.divider()
    
    # Keywords Section
    st.markdown("## 🏷️ Keywords Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Your Top Keywords")
        st.write("Keywords found in your resume:")
        
        # Create keyword display
        keyword_cols = st.columns(3)
        for idx, (keyword, freq) in enumerate(resume_keywords[:9]):
            with keyword_cols[idx % 3]:
                st.info(f"**{keyword}**")
    
    with col2:
        st.markdown("### 🔵 Job Requirements Keywords")
        st.write("Top keywords from job description:")
        
        # Create keyword display
        keyword_cols = st.columns(3)
        for idx, (keyword, freq) in enumerate(job_keywords[:9]):
            with keyword_cols[idx % 3]:
                st.info(f"**{keyword}**")
    
    st.divider()
    
    # Missing Keywords
    if missing_keywords:
        st.markdown("## 📌 Missing Keywords (Add These!)")
        st.write("Keywords from the job description that are **NOT** in your resume:")
        
        missing_cols = st.columns(min(5, len(missing_keywords)))
        for idx, (keyword, _) in enumerate(missing_keywords[:5]):
            with missing_cols[idx % 5]:
                st.error(f"❌ {keyword}")
        
        st.divider()
    
    # Improvement Suggestions
    st.markdown("## 💡 Improvement Suggestions")
    
    for idx, suggestion in enumerate(suggestions, 1):
        col1, col2 = st.columns([0.15, 0.85])
        
        with col1:
            st.markdown(f"## {suggestion['emoji']}")
        
        with col2:
            if suggestion['priority'] == 'Critical':
                st.error(f"**{suggestion['title']}** — {suggestion['description']}")
            elif suggestion['priority'] == 'High':
                st.warning(f"**{suggestion['title']}** — {suggestion['description']}")
            elif suggestion['priority'] == 'Medium':
                st.info(f"**{suggestion['title']}** — {suggestion['description']}")
            else:
                st.success(f"**{suggestion['title']}** — {suggestion['description']}")
    
    st.divider()
    
    # Action Plan
    st.markdown("## ✅ Action Plan")
    
    action_steps = [
        ("1️⃣", "Review the missing keywords above"),
        ("2️⃣", "Incorporate them naturally into your resume"),
        ("3️⃣", "Update your professional summary or key sections"),
        ("4️⃣", "Add more specific examples and achievements"),
        ("5️⃣", "Re-analyze to track improvements"),
        ("6️⃣", "Aim for 70%+ match score before applying"),
    ]
    
    for emoji, step in action_steps:
        st.write(f"{emoji} {step}")
    
    st.divider()
    
    # Summary
    st.markdown("## 📈 Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.metric("Match Score", f"{similarity_score}%", delta="Target: 70%+")
    
    with summary_col2:
        if similarity_score >= 70:
            st.metric("Status", "Ready to Apply! ✅")
        elif similarity_score >= 50:
            st.metric("Status", "Needs Polish 🔨")
        else:
            st.metric("Status", "Needs Work 🛠️")
    
    # Export suggestions (optional)
    st.markdown("---")
    
    # Create a summary text for download
    summary_text = f"""
RESUME JOB MATCH ANALYSIS
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

JOB INFORMATION
---------------
Position: {job_title if job_title else 'Not specified'}
Match Score: {similarity_score}%

RESUME STATISTICS
-----------------
Total Words: {len(resume_text.split())}
Top Keywords: {', '.join([w for w, _ in resume_keywords[:5]])}

MISSING KEYWORDS (Add these!)
-----------------------------
{', '.join([w for w, _ in missing_keywords[:10]])}

RECOMMENDATIONS
---------------
"""
    
    for idx, suggestion in enumerate(suggestions, 1):
        summary_text += f"\n{idx}. {suggestion['emoji']} {suggestion['title']}\n   {suggestion['description']}\n"
    
    st.download_button(
        label="📥 Download Analysis Summary",
        data=summary_text,
        file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

else:
    # Welcome message when no analysis is done
    st.divider()
    st.markdown("### 👋 Welcome!")
    st.write("""
    Upload your resume and job description to get started. 
    Our tool will analyze how well they match and provide actionable suggestions to improve your resume.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📄 Step 1")
        st.write("Upload your resume in PDF format")
    
    with col2:
        st.markdown("#### 📋 Step 2")
        st.write("Paste the complete job description")
    
    with col3:
        st.markdown("#### 🔍 Step 3")
        st.write("Click Analyze Match to see results")