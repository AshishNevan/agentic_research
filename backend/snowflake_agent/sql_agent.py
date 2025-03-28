from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import os
import re

from snowflake_agent.LLMManager import LLMManager

load_dotenv("../../.env")

# Generate a query based on user input
PROMPT = """
You are an AI assistant that generates SQL queries based on user questions, database schema, and unique nouns found in the already existing relevant tables. Your task is to generate a valid SQL query to answer the user's question and return the query output exactly as it is.

Consider the following:
- Analyze the database schema to identify relevant tables and columns.
- Use appropriate SQL functions and clauses to ensure the query is efficient and accurate.
- If there is not enough information to write a SQL query, respond with "NOT_ENOUGH_INFO".
- If the user's question is not related to the database, respond with "NOT_RELEVANT".
- Provide explanations for any assumptions made during query generation.

Generate the SQL query for the following question:
"""


class SnowflakeAgent:
    def __init__(
        self, db_uri, model_provider="openai", model_name="gpt-3.5-turbo", temperature=0
    ):
        self.db = SQLDatabase.from_uri(db_uri)

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        self.llm = init_chat_model(
            model_name, model_provider=model_provider, temperature=temperature
        )
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        self.llm_manager = LLMManager()

    def generate_query_result(self, user_query):
        result = self.agent_executor.invoke(
            {"input": f"System:{PROMPT}, Question:{user_query}"}
        )
        return result

    def choose_visualization(self, user_query, results):
        """
        Choose a visualization type based on the returned results
        """
        # results = self.generate_query_result(user_query)
        if results == "NOT_ENOUGH_INFO":
            return {
                "visualization": None,
                "visualization_reasoning": "Not enough information to generate a visualization.",
            }
        if results == "NOT_RELEVANT":
            return {
                "visualization": None,
                "visualization_reasoning": "No visualization needed for an irrelevant question.",
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an AI assistant that recommends appropriate data visualizations. Based on the user's question, SQL query, and query results, suggest the most suitable type of graph or chart to visualize the data. If no visualization is appropriate, indicate that.

Available chart types and their use cases:
- Bar Graphs: Best for comparing categorical data or showing changes over time when categories are discrete and the number of categories is more than 2. Use for questions like "What are the valuation measures for each year?" or "How does the population of cities compare?" or "What percentage of each city is male?"
- Horizontal Bar Graphs: Best for comparing categorical data or showing changes over time when the number of categories is small or the disparity between categories is large. Use for questions like "Show the revenue of A and B?" or "How does the population of 2 cities compare?" or "How many men and women got promoted?" or "What percentage of men and what percentage of women got promoted?" when the disparity between categories is large.
- Scatter Plots: Useful for identifying relationships or correlations between two numerical variables or plotting distributions of data. Best used when both x axis and y axis are continuous. Use for questions like "Plot a distribution of the fares (where the x axis is the fare and the y axis is the count of people who paid that fare)" or "Is there a relationship between advertising spend and sales?" or "How do height and weight correlate in the dataset?"
- Pie Charts: Ideal for showing proportions or percentages within a whole. Use for questions like "What is the market share distribution among different companies?" or "What percentage of the total revenue comes from each product?"
- Line Graphs: Best for showing trends and distributions over time. Best used when both x axis and y axis are continuous. Used for questions like "How have website visits changed over the year?" or "What is the trend in temperature over the past decade?"
- Area Charts: Useful for showing cumulative totals over time or comparing multiple categories. Use for questions like "What is the cumulative sales over the year?" or "How do different categories contribute to the total sales over time?"
- Bubble Charts: Useful for visualizing three dimensions of data, where size represents a third variable. Use for questions like "How do sales, profit, and market share compare across products?"

Consider these types of questions when recommending a visualization:
1. Aggregations and Summarizations (e.g., "What is the average revenue by month?" - Line Graph)
2. Comparisons (e.g., "Compare the sales figures of Product A and Product B over the last year." - Line or Column Graph)
3. Plotting Distributions (e.g., "Plot a distribution of the age of users" - Scatter Plot)
4. Trends Over Time (e.g., "What is the trend in the number of active users over the past year?" - Line Graph)
5. Proportions (e.g., "What is the market share of the products?" - Pie Chart)
6. Correlations (e.g., "Is there a correlation between marketing spend and revenue?" - Scatter Plot)
7. Cumulative Totals (e.g., "What is the cumulative sales over the year?" - Area Chart)
8. Multi-dimensional Comparisons (e.g., "How do sales, profit, and market share compare across products?" - Bubble Chart)

Provide your response in the following format:
Recommended Visualization: [Chart type or "None"]. ONLY use the following names: bar, horizontal_bar, line, pie, scatter, area, bubble, none
Reason: [Brief explanation for your recommendation]
""",
                ),
                (
                    "human",
                    """
User question: {user_query}
Query results: {results}

Recommend a visualization:""",
                ),
            ]
        )

        response = self.llm_manager.invoke(
            prompt, user_query=user_query, results=results
        )
        lines = response.split("\n")
        visualization = lines[0].split(":")[1].strip().lower()
        reasoning = lines[1].split(":")[1].strip()

        return {"visualization": visualization, "visualization_reasoning": reasoning}

    def generate_visualization_code(self, user_query):
        """
        Use LLM to generate visualization code based on the data and the visualization type
        """
        results = self.generate_query_result(user_query)
        visualization_info = self.choose_visualization(user_query, results)
        if visualization_info["visualization"] == "none":
            return {
                "visualization_code": None,
                "visualization_reasoning": "No visualization needed for an irrelevant question.",
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            Generate Python code using matplotlib to create a {visualization_type} chart for the following data:
    {results}
    
    The code should:
    1. Create a visually appealing {visualization_type} chart
    2. Include proper formatting (titles, labels, etc.)
    3. Use appropriate colors and styling
    4. Format numbers with commas for readability
    5. Return only the raw Python code without any Markdown formatting, explanations, or comments
    6. Do not wrap the code in triple backticks or any other formatting
    7. Use plt.savefig() to save the image as a PNG file in the current directory
    8. Do not include plt.show() in the code
    """,
                ),
                (
                    "human",
                    """
            Data: {results}
            Visualization type: {visualization_type}
            Generate Python code:
            """,
                ),
            ]
        )

        response = self.llm_manager.invoke(
            prompt,
            results=results,
            visualization_type=visualization_info["visualization"],
        )
        # Sanitize any remaining markdown formatting
        lines = response.split("\n")
        filtered_lines = [
            line
            for line in lines
            if line.strip() not in ("```python", "```") and "plt.show()" not in line
        ]
        sanitized_code = "\n".join(filtered_lines)
        return {
            "visualization_code": sanitized_code,
            "visualization_reasoning": visualization_info["visualization_reasoning"],
        }


# Create an instance of SnowflakeAgent
if __name__ == "__main__":
    agent = SnowflakeAgent(os.environ.get("SNOWFLAKE_URI"))
    query = "How is the P/E ratio trending over the past 4 quarters?"
    generated_code = agent.generate_visualization_code(query)["visualization_code"]
    print(generated_code)
    exec(generated_code)
