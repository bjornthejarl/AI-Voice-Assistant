"""
Smart Tools for Sarah Voice Assistant
Provides various utilities that the AI can call to perform real-world actions.
"""

import os
import subprocess
import threading
import time
from datetime import datetime
import json

# Try to import optional dependencies
try:
    from duckduckgo_search import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    print("⚠️ Web search unavailable. Install with: pip install duckduckgo-search")

try:
    import requests
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False


# ============== TIME & DATE ==============

def get_current_time() -> dict:
    """Get the current time and date."""
    now = datetime.now()
    return {
        "time": now.strftime("%I:%M %p"),
        "date": now.strftime("%A, %B %d, %Y"),
        "day_of_week": now.strftime("%A"),
        "formatted": now.strftime("%I:%M %p on %A, %B %d, %Y")
    }


# ============== WEATHER ==============

def get_weather(city: str, api_key: str = None) -> dict:
    """Get weather for a city using OpenWeatherMap API."""
    if not WEATHER_AVAILABLE:
        return {"error": "Weather functionality requires the requests library"}
    
    if not api_key:
        return {"error": "Weather API key not configured. Add OPENWEATHERMAP_API_KEY to config.py"}
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            return {
                "city": data["name"],
                "temperature_celsius": round(data["main"]["temp"]),
                "temperature_fahrenheit": round(data["main"]["temp"] * 9/5 + 32),
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "formatted": f"{data['name']}: {round(data['main']['temp'])}°C ({round(data['main']['temp'] * 9/5 + 32)}°F), {data['weather'][0]['description']}"
            }
        else:
            return {"error": f"Could not find weather for {city}"}
    except Exception as e:
        return {"error": f"Weather lookup failed: {str(e)}"}


# ============== WEB SEARCH ==============

# Free public SearXNG instances (no API key needed)
SEARXNG_INSTANCES = [
    "https://search.bus-hit.me",
    "https://searx.be",
    "https://search.ononoki.org",
    "https://searx.tiekoetter.com",
]

def web_search(query: str, max_results: int = 3) -> dict:
    """Search the web using free SearXNG public instances."""
    import requests
    import time as _time
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    # Try each instance until one works
    for instance in SEARXNG_INSTANCES:
        try:
            response = requests.get(
                f"{instance}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                },
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])[:max_results]
                
                if results:
                    formatted_results = []
                    for r in results:
                        formatted_results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("content", ""),
                            "url": r.get("url", "")
                        })
                    
                    summary = results[0].get("content", "")[:150] if results else "No results."
                    
                    return {
                        "results": formatted_results,
                        "summary": summary
                    }
                    
        except Exception:
            _time.sleep(0.5)
            continue
    
    # All instances failed
    return {"error": "Search unavailable right now. Try asking me directly."}


# ============== OPEN APPLICATIONS ==============

# Common Windows applications and their paths/commands
WINDOWS_APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "command prompt": "cmd.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "control panel": "control.exe",
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "spotify": "spotify",
    "discord": "discord",
    "steam": "steam",
    "vscode": "code",
    "visual studio code": "code",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "outlook": "outlook",
}

def open_application(app_name: str) -> dict:
    """Open an application on Windows."""
    app_lower = app_name.lower().strip()
    
    # Check if it's a known app
    if app_lower in WINDOWS_APPS:
        app_command = WINDOWS_APPS[app_lower]
    else:
        # Try to open it directly
        app_command = app_lower
    
    try:
        # Use start command to open apps without blocking
        if app_command.startswith("ms-"):
            # Windows URI scheme
            os.system(f'start "" "{app_command}"')
        else:
            subprocess.Popen(
                ["cmd", "/c", "start", "", app_command],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        return {"success": True, "message": f"Opening {app_name}"}
    except Exception as e:
        return {"success": False, "error": f"Could not open {app_name}: {str(e)}"}


# ============== REMINDERS ==============

active_reminders = []

def set_reminder(message: str, minutes: float) -> dict:
    """Set a reminder that will trigger after the specified minutes."""
    
    def reminder_callback():
        time.sleep(minutes * 60)
        print(f"\n🔔 REMINDER: {message}")
        # We'll handle the actual notification in the main assistant
        
    reminder_thread = threading.Thread(target=reminder_callback, daemon=True)
    reminder_thread.start()
    
    reminder_info = {
        "message": message,
        "minutes": minutes,
        "trigger_time": (datetime.now().timestamp() + minutes * 60)
    }
    active_reminders.append(reminder_info)
    
    return {
        "success": True, 
        "message": f"Reminder set for {minutes} minute{'s' if minutes != 1 else ''}: {message}"
    }



# ============== TOOL DEFINITIONS FOR DEEPSEEK ==============

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time and date. Use this when the user asks about the time or date.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. Use this when the user asks about weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Use this when the user wants to look something up or needs current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Open an application on the user's computer. Use this when the user wants to open a program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "The name of the application to open (e.g., 'notepad', 'chrome', 'calculator')"
                    }
                },
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder that will alert the user after a specified number of minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The reminder message"
                    },
                    "minutes": {
                        "type": "number",
                        "description": "Number of minutes until the reminder"
                    }
                },
                "required": ["message", "minutes"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: dict, weather_api_key: str = None) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if tool_name == "get_current_time":
            result = get_current_time()
        elif tool_name == "get_weather":
            result = get_weather(arguments.get("city", ""), weather_api_key)
        elif tool_name == "web_search":
            result = web_search(arguments.get("query", ""))
        elif tool_name == "open_application":
            result = open_application(arguments.get("app_name", ""))
        elif tool_name == "set_reminder":
            result = set_reminder(
                arguments.get("message", "Reminder"),
                arguments.get("minutes", 5)
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
