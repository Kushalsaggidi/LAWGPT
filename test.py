import aiohttp
import asyncio
import os

API_KEY = "AIzaSyCAEEIT9lc5WxLAyq_N4SaKouAggntDxC4"  # replace or use env var
MODEL = "gemini-2.5-flash-lite"
MAX_QUOTA = 1000  # free tier per day (from docs)

class QuotaTracker:
    def __init__(self, limit=MAX_QUOTA):
        self.limit = limit
        self.used = 0

    def remaining(self):
        return self.limit - self.used

quota = QuotaTracker()

async def call_gemini(session, prompt):
    global quota
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with session.post(url, headers=headers, json=payload) as resp:
        data = await resp.json()
        if resp.status == 200:
            quota.used += 1
            return data
        elif resp.status == 429:
            print("⚠️ Quota exhausted (429). Stop sending more requests.")
        elif resp.status == 401:
            print("❌ Invalid API key.")
        else:
            print(f"⚠️ Unexpected error {resp.status}: {data}")
        return None

async def main():
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            print(f"\n🔄 Sending request {i+1}...")
            resp = await call_gemini(session, "Quick test.")
            if not resp:
                break
            print(f"✅ Response received. Used: {quota.used}, Remaining: {quota.remaining()}")

if __name__ == "__main__":
    asyncio.run(main())
