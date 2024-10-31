import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import io
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client using OpenRouter API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def fetch_webpage(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลจากเว็บไซต์: {e}")
        return None

def extract_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    title = soup.title.string.strip() if soup.title else "ไม่พบชื่อเรื่อง"
    h1 = soup.find('h1').text.strip() if soup.find('h1') else "ไม่พบ H1"
    h2s = [h2.text.strip() for h2 in soup.find_all('h2')]
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    meta_desc = meta_desc['content'].strip() if meta_desc else "ไม่พบคำอธิบายเมตา"
    content = soup.get_text(separator='\n', strip=True)
    
    return {
        'title': title,
        'h1': h1,
        'h2s': h2s,
        'meta_description': meta_desc,
        'content': content
    }

def extract_keywords(text, num_keywords=5):
    prompt = f"""
    Analyze the following text elements from a webpage and extract the top {num_keywords} most relevant keywords or key phrases. 
    Focus primarily on the title, H1, H2s, and meta description. Consider the semantic meaning and context of the words.
    Provide the keywords in order of relevance.
    
    Title: {text['title']}
    H1: {text['h1']}
    H2s: {', '.join(text['h2s'])}
    Meta Description: {text['meta_description']}
    
    เพิ่มเนื้อหา: {text['content'][:500]}  # Limiting to 500 characters
    
    คีย์เวิร์ด:
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts keywords from webpage content. The content may be in Thai or English, or a mix of both. Provide the most relevant keywords or phrases, which can be in either language depending on the content."},
            {"role": "user", "content": prompt}
        ]
    )
    
    keywords = response.choices[0].message.content.strip().split('\n')
    return [kw.strip() for kw in keywords if kw.strip()][:num_keywords]

def parse_keywords_input(keyword_input):
    """
    Parse the keyword input which can be in 'Keyword,Volume' or 'Keyword\nVolume' format.
    """
    lines = keyword_input.strip().split('\n')
    keywords = []
    volumes = []
    
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Determine the format
    if ',' in lines[0]:
        # Assume 'Keyword,Volume' format
        for line in lines:
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    keyword = parts[0].strip()
                    volume = parts[1].strip()
                    keywords.append(keyword)
                    volumes.append(volume)
    else:
        # Assume 'Keyword\nVolume' format
        for i in range(0, len(lines), 2):
            if i+1 < len(lines):
                keyword = lines[i].strip()
                volume = lines[i+1].strip()
                keywords.append(keyword)
                volumes.append(volume)
    
    # Clean and convert volumes
    clean_volumes = []
    for vol in volumes:
        # Handle ranges like '0–10' by taking the maximum value
        match = re.search(r'(\d+)', vol)
        if match:
            clean_volumes.append(int(match.group(1)))
        else:
            clean_volumes.append(0)  # Default to 0 if no number found
    
    # Create DataFrame
    df = pd.DataFrame({
        'Keyword': keywords,
        'Volume': clean_volumes
    })
    
    # Remove any rows with empty keywords
    df = df[df['Keyword'] != '']
    
    return df

def generate_seo_recommendations(current_elements, keywords_df):
    main_keyword = keywords_df.iloc[0]['Keyword']
    other_keywords = keywords_df.iloc[1:4]['Keyword'].tolist()
    faq_keywords = keywords_df.iloc[4:]['Keyword'].tolist()
    
    prompt = f"""
    As an SEO expert, analyse the current webpage elements and the new set of target keywords.
    Provide specific recommendations for optimising the Title, H1, H2s, Meta Description, and additional content to better align with the new main keyword.
    Ensure that each H2 is clear, distinct, and covers a unique aspect related to the main search intent.
    Avoid duplicating H2s and integrate keywords naturally, preferably having keywords early in each element such as Title, H1, H2, Meta Description. Each H2 should be in separate paragraph.
    Preserve existing promotional content such as vpn softwares, antivirus softwares, AI softwares,presale, new, and meme tokens naturally within the content.
    Include a comprehensive FAQ section using the provided FAQ keywords, ensure they are not duplicating with H2s
    You are generating SEO recommendations for Thai content Editor, all recommendations should be in Thai, but keywords and examples of keyword usage can be in English or Thai as appropriate.
    
    The main keyword is "{main_keyword}". Ensure it's used in the Title, H1, and Meta Description.
    Use the remaining keywords naturally throughout the content, placing FAQ-related keywords in the FAQ section.

    Main Keyword: {main_keyword}
    Other Main Keywords for H2s: {', '.join(other_keywords)}
    FAQ Keywords: {', '.join(faq_keywords)}
    
    Current Content Structure:
    Title: {current_elements['title']}
    H1: {current_elements['h1']}
    H2s: {', '.join(current_elements['h2s'])}
    Meta Description: {current_elements['meta_description']}
    
    Provide your recommendations in the following format:
    SEO Recommendations
    Title: [Optimized Title]
    H1 ที่แนะนำ: [Optimized H1]
    Meta Description:
    [Optimized Meta Description]
    โครงสร้างเนื้อหา H2 ที่แนะนำ:
    H2 - [New H2 based on other keywords]
    H2 - [New H2 based on other keywords]
    H2 - [New H2 based on other keywords]
    ส่วน FAQ ที่แนะนำเพิ่มเติม:
    [FAQ Question 1]?
    [FAQ Question 2]?
    [FAQ Question 3]?
    [FAQ Question 4]?

    คำแนะนำ:
    """

    response = client.chat.completions.create(
        model="openai/gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an SEO expert providing concise recommendations for content optimization based on the provided keywords and content structure. Your output should be in Thai."},
            {"role": "user", "content": prompt}
        ]
    )
    
    recommendations = response.choices[0].message.content.strip()
    return recommendations

def main():
    st.title("Webpage Content Optimizer")
    
    url = st.text_input("Enter a webpage URL:", value="https://th.cryptonews.com/tech/best-vpn-for-crypto/")
    
    if 'webpage_elements' not in st.session_state:
        st.session_state['webpage_elements'] = None
    
    if 'keywords_df' not in st.session_state:
        st.session_state['keywords_df'] = None
    
    if st.button("Extract and Analyze"):
        if url:
            html_content = fetch_webpage(url)
            if html_content:
                extracted_content = extract_content(html_content)
                
                # Store HTML elements in session state
                st.session_state['webpage_elements'] = extracted_content
                
                st.subheader("Webpage Title")
                st.write(extracted_content['title'])
                
                st.subheader("H1")
                st.write(extracted_content['h1'])
                
                st.subheader("H2s")
                st.write(", ".join(extracted_content['h2s']))
                
                st.subheader("Meta Description")
                st.write(extracted_content['meta_description'])
                
                st.subheader("Extracted Content")
                st.text_area("Content", extracted_content['content'], height=200)
                
                keywords = extract_keywords(extracted_content)
                
                st.subheader("Top 5 Keywords/Phrases")
                for i, keyword in enumerate(keywords, 1):
                    st.write(f"{i}. {keyword}")
                
                st.success("Content extracted and analyzed. You can now proceed to enter your keywords for optimization.")
        else:
            st.warning("Please enter a valid URL.")
    
    st.subheader("Enter Keywords")
    keyword_input = st.text_area(
        "Enter keywords and search volumes (format: 'Keyword,Volume' or 'Keyword\\nVolume'):",
        value="""vpn แนะนำ
100	

vpn ที่ดีที่สุด
60	

vpn ในโทรศัพท์คืออะไร
1.5K	

vpn ปิดยังไง
300	

ปิด vpn iphone
200	

free vpn for iphone
200	

vpn ในไอโฟน คืออะไร
150	

vpn ไทย แรงๆ
150	

vpn ในไอโฟนคืออะไร
150	

vpn ราคาถูก
100	

vpn คืออะไร มีประโยชน์อย่างไร
70	

vpn ใช้ยังไง
70	

express vpn ดีไหม
70	

express vpn ราคา
60""",
        height=300
    )
    
    if st.button("Process Keywords"):
        try:
            # Process the pasted keywords
            df = parse_keywords_input(keyword_input)
            
            # Ensure 'Keyword' and 'Volume' columns exist
            if 'Keyword' not in df.columns or 'Volume' not in df.columns:
                st.error("Please ensure the input has 'Keyword' and 'Volume' columns.")
                return
            
            st.session_state['keywords_df'] = df
            st.write(df)
            
            st.success("Keywords processed. You can now generate SEO recommendations.")
        
        except Exception as e:
            st.error(f"Error processing the keywords: {e}")
            st.error("Please ensure the input is in the correct format: 'Keyword,Volume' or 'Keyword\\nVolume' for each entry.")
    
    if st.button("Generate SEO Recommendations"):
        if st.session_state['webpage_elements'] is not None and st.session_state['keywords_df'] is not None:
            recommendations = generate_seo_recommendations(st.session_state['webpage_elements'], st.session_state['keywords_df'])
            st.subheader("SEO Recommendations")
            
            # Display the important statement
            st.markdown("**สำคัญมาก:** คำแนะนำการใช้คีย์เวิร์ดในตำแหน่งสำคัญมีไว้ในรายละเอียดในบรีฟนี้แล้ว เช่น Title, Intro, Meta Description, H1, H2 ต่างๆ. แต่ถ้าจะเพิ่มคีย์เวิร์ด ให้ใช้อย่างเป็นธรรมชาติประมาณไม่เกิน 20 ครั้งต่อ 5000 คำ และใช้เทคนิคการปรับเปลี่ยนคำ (synonyms or variations) แทน")
            
            # Display the keyword table
            st.subheader("Keyword Table")
            st.table(st.session_state['keywords_df'][['Keyword', 'Volume']].rename(columns={'Volume': 'Search Volume'}))
            
            # Display the recommendations
            st.subheader("Detailed Recommendations")
            st.write(recommendations)
        else:
            st.warning("Please extract webpage content and process keywords first.")

if __name__ == "__main__":
    main()
