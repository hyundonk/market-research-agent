from strands import Agent
from strands.models import BedrockModel
from strands.tools import tool
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import os
import pandas as pd
import time
import re
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", #"us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    temperature=0.1,
)

# Evaluation agent for GameLift eligibility assessment
evaluation_model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",# "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    temperature=0.0,
)

evaluation_prompt = (
    "You are a GameLift Eligibility Evaluator. "
    "Analyze the game information and determine if it's suitable for Amazon GameLift.\n\n"
    "GameLift is ideal for:\n"
    "- Session-based multiplayer games (matches, rounds, battles)\n"
    "- PvP competitive games (FPS, TPS, MOBA, Battle Royale)\n"
    "- Games with matchmaking and lobbies\n"
    "- Real-time multiplayer with defined game sessions\n\n"
    "GameLift is NOT suitable for:\n"
    "- MMORPGs with persistent worlds\n"
    "- Single-player only games\n"
    "- Asynchronous multiplayer\n"
    "- Open world games without session boundaries\n\n"
    "Extract the following information from the search results and respond in this exact format:\n"
    "ELIGIBLE: [Yes/No/Maybe]\n"
    "CONFIDENCE: [High/Medium/Low]\n"
    "REASON: [Brief explanation]\n"
    "DEVELOPER: [developer company if found, otherwise 'Unknown']\n"
    "PUBLISHER: [publisher company if found, otherwise 'Unknown']\n"
    "GENRE: [game genre if found, otherwise 'Unknown']\n"
    "TECHNICAL: [engine/platform info if found, otherwise 'Unknown']"
)

system_prompt = (
    "You are a Korean Gaming Market Research Agent. "
    "Extract game information from Korean news search results and respond in this exact format:\n"
    "GAMELIFT_ELIGIBLE: [Yes/No - if game supports session-based online multiplayer with PvP modes suitable for Amazon GameLift]\n"
    "DEVELOPER: [developer company if found, otherwise 'Unknown']\n"
    "PUBLISHER: [publisher company if found, otherwise 'Unknown']\n"
    "GENRE: [game genre if found, otherwise 'Unknown']\n"
    "TECHNICAL: [engine/platform info if found, otherwise 'Unknown']"
)

@tool
def search_korean_news(query: str, display: int = 10) -> str:
    """Search Korean news articles using Naver search."""
    print(f"Korean News Query: {query}")
    async def _search():
        try:
            env = os.environ.copy()
            
            server_params = StdioServerParameters(
                command="node",
                args=["./mcps/naver-search-mcp/dist/src/index.js"],
                env=env
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("search_webkr", {"query": query, "display": display})
                    
                    if result and hasattr(result, 'content') and result.content:
                        json_text = result.content[0].text
                        data = json.loads(json_text)
                        
                        formatted_results = ""
                        for i, item in enumerate(data.get('items', []), 1):
                            title = item.get('title', '').replace('<b>', '').replace('</b>', '')
                            description = item.get('description', '').replace('<b>', '').replace('</b>', '')
                            formatted_results += f"{title} - {description}\n"
                        
                        # Log search results
                        with open('search_text.log', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"Korean News Search: {query}\n")
                            log_file.write(formatted_results)
                            log_file.write("\n")
                        
                        return formatted_results
                    else:
                        return "No results found"
        except Exception as e:
            return f"Search error: {str(e)}"
    
    try:
        return asyncio.run(_search())
    except Exception as e:
        return f"Connection error: {str(e)}"

@tool
def search_english_news(query: str, display: int = 10) -> str:
    """Search English news articles using Google search MCP."""
    print(f"English News Query: {query}")
    async def _search():
        try:
            server_params = StdioServerParameters(
                command="python",
                args=["./mcps/google_search/google_search.py"]
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("google_search", {"query": query, "num_results": display})
                    
                    if result and hasattr(result, 'content') and result.content:
                        search_results = result.content[0].text
                        
                        # Log search results
                        with open('search_text.log', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"English News Search: {query}\n")
                            log_file.write(search_results)
                            log_file.write("\n")
                        
                        return search_results
                    else:
                        return "No results found"
        except Exception as e:
            return f"Search error: {str(e)}"
    
    try:
        return asyncio.run(_search())
    except Exception as e:
        return f"Connection error: {str(e)}"

def evaluate_with_ai(search_text: str, game_title: str) -> dict:
    """Use AI agent to evaluate GameLift eligibility and extract game information."""
    game_info = {
        'Game Title': game_title,
        'AI Evaluation': 'Unknown',
        'AI Confidence': 'Unknown',
        'AI Reason': 'Not evaluated',
        'Developer': 'Unknown', 
        'Publisher': 'Unknown',
        'Genre': 'Unknown',
        'Technical Description': 'Unknown',
        'Additional Information': 'No additional info found'
    }
    
    if not search_text or "No results found" in search_text:
        return game_info
    
    # Extract additional information
    additional_info = []
    if '출시' in search_text or 'release' in search_text.lower():
        additional_info.append('Release information available')
    if '업데이트' in search_text or 'update' in search_text.lower():
        additional_info.append('Recent updates mentioned')
    if '플레이어' in search_text or 'player' in search_text.lower():
        additional_info.append('Player information available')
    
    if additional_info:
        game_info['Additional Information'] = '; '.join(additional_info)
    
    try:
        eval_agent = Agent(
            system_prompt=evaluation_prompt,
            model=evaluation_model
        )
        
        eval_prompt = f"""
Game: {game_title}

Search Results Summary:
{search_text[:2000]}

Based on this information, evaluate if this game is suitable for Amazon GameLift and extract the requested information.
"""
        
        response = eval_agent(eval_prompt)
        response_text = str(response.content) if hasattr(response, 'content') else str(response)
        
        # Parse AI response
        if 'ELIGIBLE:' in response_text:
            eligible_match = re.search(r'ELIGIBLE:\s*(\w+)', response_text)
            confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', response_text)
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|DEVELOPER:|$)', response_text)
            developer_match = re.search(r'DEVELOPER:\s*(.+?)(?:\n|PUBLISHER:|$)', response_text)
            publisher_match = re.search(r'PUBLISHER:\s*(.+?)(?:\n|GENRE:|$)', response_text)
            genre_match = re.search(r'GENRE:\s*(.+?)(?:\n|TECHNICAL:|$)', response_text)
            technical_match = re.search(r'TECHNICAL:\s*(.+?)(?:\n|$)', response_text)
            
            if eligible_match:
                game_info['AI Evaluation'] = eligible_match.group(1)
            if confidence_match:
                game_info['AI Confidence'] = confidence_match.group(1)
            if reason_match:
                game_info['AI Reason'] = reason_match.group(1).strip()
            if developer_match:
                game_info['Developer'] = developer_match.group(1).strip()
            if publisher_match:
                game_info['Publisher'] = publisher_match.group(1).strip()
            if genre_match:
                game_info['Genre'] = genre_match.group(1).strip()
            if technical_match:
                game_info['Technical Description'] = technical_match.group(1).strip()
        
    except Exception as e:
        game_info['AI Reason'] = f'Evaluation error: {str(e)}'
    
    return game_info

def process_games_production(input_file):
    """Process all games from CSV and create final output."""
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        print(f"Found {len(df)} games to process")
        
        # Create an agent
        agent = Agent(
            system_prompt=system_prompt,
            model=bedrock_model,
            tools=[search_korean_news, search_english_news]
        )
        
        games_data = []
        
        # Process all games
        for i, row in df.iterrows():
            game_title = row['App Name']
            company_name = row.get('Company Name', '')
            print(f"Processing {i+1}/{len(df)}: {game_title}")
            
            try:
                # Search for game information in both Korean and English
                korean_query = f"{game_title} {company_name}".strip()
                english_query = f"{game_title} {company_name}".strip()
                search_prompt = f"Search Korean news for '{korean_query}' and English news for '{english_query}' and return combined search results."
                
                # Log game processing start
                with open('search_text.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n{'='*50}\n")
                    log_file.write(f"Game: {game_title}\n")
                    log_file.write(f"{'='*50}\n")
                
                response = agent(search_prompt)
                
                # Convert response to string
                if hasattr(response, 'content'):
                    search_text = str(response.content)
                else:
                    search_text = str(response)
                
                # Log final response
                with open('search_text.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Agent Response:\n{search_text}\n")
                    log_file.write(f"{'='*50}\n\n")
                
                # Extract information and AI evaluation
                game_info = evaluate_with_ai(search_text, game_title)
                
                games_data.append(game_info)
                
                print(f"  → AI: {game_info['AI Evaluation']} ({game_info['AI Confidence']})")
                
                # Save progress every 50 games
                if (i + 1) % 50 == 0:
                    df_temp = pd.DataFrame(games_data)
                    df_temp.to_csv(f'games_progress_{i+1}.csv', index=False, encoding='utf-8-sig')
                    print(f"  Saved progress: {i+1} games")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                # Add empty entry for failed games
                games_data.append({
                    'Game Title': game_title,
                    'AI Evaluation': 'Unknown',
                    'AI Confidence': 'Unknown',
                    'AI Reason': 'Search failed',
                    'Developer': 'Unknown', 
                    'Publisher': 'Unknown',
                    'Genre': 'Unknown',
                    'Technical Description': 'Unknown',
                    'Additional Information': 'Search failed'
                })
        
        # Create the final CSV
        df_output = pd.DataFrame(games_data)
        df_output.to_csv('data-ai_2025_games.csv', index=False, encoding='utf-8-sig')
        
        print(f"\nSuccessfully created data-ai_2025_games.csv with {len(games_data)} games")
        
        # Print summary
        print("\n=== SUMMARY ===")
        developers = df_output['Developer'].value_counts()
        print(f"Top developers found: {dict(developers.head())}")
        
        return games_data
        
    except Exception as e:
        print(f"Error processing games: {str(e)}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python production_game_agent.py <input_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    process_games_production(input_file)
