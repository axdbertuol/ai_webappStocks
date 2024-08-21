import json
import os
from datetime import datetime

import streamlit as st
import yfinance as yf
from crewai import Agent, Crew, Process, Task
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI


def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock


yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stock prices from Yahoo Finance",
    func=lambda ticket: fetch_stock_price(ticket),
)


os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo")


stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyze trends",
    backstory="""
        You're a highly experienced price analyst of specialized stock prices, 
        and you should be able to make predictions aboput its future prices.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
    tools=[yahoo_finance_tool],
)


getStockPrice = Task(
    description="Analyze stock {ticket} price history and create trend analysis of up, down or sideways prices.",
    expected_output="""
        Specify the current trend stock price - up, down or sideways.
        eg. stock='APPL, price UP'
    """,
    agent=stockPriceAnalyst,
)


search_tool = DuckDuckGoSearchResults(
    backend="news",
    num_results=10,
)


newsAnalyst = Agent(
    role="Senior stock research Analyst",
    goal="""
        Summarize news articles related to the stock {ticket} company. 
        Specify the current trend - up, down or sideways according to the news context.
        For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.
    """,
    backstory="""
        You're a highly experienced market trends analyst and their news.
        You have tracked assets for more than ten years.
        You're also a master level analyst of traditional market stock exchanges and have a deep understanding of human psychology.
        You must be able to understand the news titles, informations and summarize articles, but you should look at them with caution and a dose of skepticism.
        You must also consider the source of news articles.
    """,
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    allow_delegation=True,
    tools=[search_tool],
)


getNews = Task(
    description=f"""
        Take the stock and always include BTC to it (if not requested).
        Use the search tool search each one individually.

        The current date is {datetime.now()}.
        The news articles should be from the last 7 days.

        Compose the results into a helpful report

        """,
    expected_output="""
        A summary of the overall market performance results for each requested asset.
        Include a feat/gread score for each asset based on the news. Use format:
        <STOCK ASSET>
        <SUMMARY BASED ON NEWS>
        <TREND PREDICTION>
        <FEAR/GREED SCORE>
        <SENTIMENT SCORE>
    """,
    agent=newsAnalyst,
)


stockAnalystReporter = Agent(
    role="Senior Stock Analyses Reporter",
    goal="""
        Analyze the trends prices and news and compile an insightful, compelling and informative report - 3 paragraphs long - based on the provided stock analysis,
          news data and price trend.
        """,
    backstory="""
        You're widely accepted as the best stock analyst in the market. 
        You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
        You understand macro factors and combine multiple theories into a single story. - eg. cycle theory and fundamental analyses.
        You're able to hold multiple opinions when analyzing anything.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
)


reportAnalyses = Task(
    description="""
        Use the stock price trends and the stock news report to create an analysis and write a compiled report about the {ticket} company
          that is brief and highlights the most important points.
        Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
        Include the previous analyses of stock trend and news summary.
    """,
    expected_output="""
        An eloquent 3 paragraphs newsletter formatted as markdown in an easy readable manner. It should contain:
        - 5 bullets executive summary
        - Introduction - set the overall picture and spike up the interest
        - Market Trends - the current trend, future trends and how they affect the company 
        - News - the most important news articles and their fear/greed score
        - Near Future Considerations - the most important points for the company and the market with up, down or sideways
        - Conclusion - the overall summary of the report
    """,
    agent=stockAnalystReporter,
    context=[getStockPrice, getNews],
)


crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystReporter],
    tasks=[getStockPrice, getNews, reportAnalyses],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15,
)


with st.sidebar:
    st.header("Type Stock ticket for research")

    with st.form(key="research_form"):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button("Run Research")

if submit_button:
    if not topic:
        st.error("Please enter a stock ticket.")
    else:
        results = crew.kickoff(inputs={"ticket": topic})
        st.subheader("Results from your research:")
        st.write(results)
