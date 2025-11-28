"""
Browser service module for headless browser operations.

Design Choice: Playwright vs Selenium
--------------------------------------
I chose Playwright over Selenium for several reasons:
1. Better async support - Playwright is designed async-first
2. Auto-wait functionality - automatically waits for elements to be ready
3. Modern architecture - faster and more reliable than Selenium
4. Built-in browser installation - no need for separate webdriver management
5. Better handling of modern JavaScript frameworks

This module handles:
- Opening URLs with JavaScript execution
- Extracting rendered DOM content
- Downloading files linked in quiz pages
- Taking screenshots if needed for debugging
"""

import asyncio
import base64
import os
import tempfile
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import urljoin, urlparse
import httpx
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class BrowserService:
    """
    Service class for headless browser operations using Playwright.
    
    This class provides methods to:
    - Navigate to URLs and render JavaScript
    - Extract page content and specific elements
    - Download files from links
    - Handle form submissions if needed
    """
    
    def __init__(self):
        """Initialize the browser service with settings."""
        self.settings = get_settings()
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._playwright = None
        
    async def __aenter__(self):
        """Async context manager entry - starts browser."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes browser."""
        await self.close()
        
    async def start(self):
        """
        Start the Playwright browser instance.
        
        Uses Chromium for best compatibility with modern web apps.
        Headless mode is configurable via settings.
        """
        self._playwright = await async_playwright().start()
        
        # Launch Chromium browser
        # Design choice: Chromium over Firefox/WebKit for best JS compatibility
        self._browser = await self._playwright.chromium.launch(
            headless=self.settings.BROWSER_HEADLESS,
            args=[
                '--disable-blink-features=AutomationControlled',  # Avoid bot detection
                '--no-sandbox',  # Required for Docker
                '--disable-dev-shm-usage',  # Overcome limited resource problems
            ]
        )
        
        # Create a browser context with reasonable viewport
        self._context = await self._browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        logger.info("Browser started successfully")
        
    async def close(self):
        """Close the browser and cleanup resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")
        
    async def get_page_content(self, url: str, wait_selector: Optional[str] = None) -> Tuple[str, str]:
        """
        Navigate to URL and get the fully rendered page content.
        
        Args:
            url: The URL to navigate to.
            wait_selector: Optional CSS selector to wait for before extracting content.
            
        Returns:
            Tuple of (HTML content, plain text content).
        """
        if not self._context:
            raise RuntimeError("Browser not started. Call start() first.")
            
        page = await self._context.new_page()
        
        try:
            # Navigate to the URL with timeout
            await page.goto(
                url,
                wait_until='networkidle',  # Wait for network to be idle
                timeout=self.settings.BROWSER_TIMEOUT
            )
            
            # Optionally wait for a specific selector
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=10000)
            
            # Small delay to ensure dynamic content loads
            await asyncio.sleep(1)
            
            # Get both HTML and text content
            html_content = await page.content()
            text_content = await page.inner_text('body')
            
            logger.debug(f"Fetched content from {url}")
            return html_content, text_content
            
        finally:
            await page.close()
            
    async def extract_quiz_data(self, url: str) -> Dict[str, Any]:
        """
        Extract quiz-specific data from a quiz page.
        
        This method navigates to the quiz URL and extracts:
        - The question/instructions text
        - Any file download links
        - The submit URL/endpoint
        - Any other relevant data attributes
        
        Args:
            url: The quiz page URL.
            
        Returns:
            Dictionary containing extracted quiz data.
        """
        if not self._context:
            raise RuntimeError("Browser not started. Call start() first.")
            
        page = await self._context.new_page()
        
        try:
            await page.goto(url, wait_until='networkidle', timeout=self.settings.BROWSER_TIMEOUT)
            await asyncio.sleep(1)  # Allow dynamic content to load
            
            # Extract the full page text for question parsing
            page_text = await page.inner_text('body')
            page_html = await page.content()
            
            # Try to find file download links (common patterns)
            # Looking for links to CSV, Excel, PDF, ZIP files
            file_links = await page.evaluate('''() => {
                const links = [];
                const anchors = document.querySelectorAll('a[href]');
                const fileExtensions = ['.csv', '.xlsx', '.xls', '.pdf', '.zip', '.json', '.txt'];
                
                anchors.forEach(a => {
                    const href = a.href;
                    if (fileExtensions.some(ext => href.toLowerCase().includes(ext))) {
                        links.push({
                            url: href,
                            text: a.innerText.trim(),
                            filename: href.split('/').pop().split('?')[0],
                            type: 'data'
                        });
                    }
                });
                
                return links;
            }''')
            
            # Also extract audio files (for questions that require audio transcription)
            audio_links = await page.evaluate('''() => {
                const audioElements = document.querySelectorAll('audio[src], audio source[src]');
                const links = [];
                
                audioElements.forEach(el => {
                    const src = el.src || el.getAttribute('src');
                    if (src) {
                        links.push({
                            url: src,
                            text: 'audio',
                            filename: src.split('/').pop().split('?')[0],
                            type: 'audio'
                        });
                    }
                });
                
                return links;
            }''')
            
            # Combine file links and audio links
            all_file_links = file_links + audio_links
            
            # Try to find submit URL - look for form action or data attributes
            submit_url = await page.evaluate(r'''() => {
                // Check for form action
                const form = document.querySelector('form');
                if (form && form.action) return form.action;
                
                // Check for data-submit-url attribute
                const submitEl = document.querySelector('[data-submit-url]');
                if (submitEl) return submitEl.getAttribute('data-submit-url');
                
                // Check for common API endpoint patterns in the page
                const pageText = document.body.innerText;
                const urlMatch = pageText.match(/submit.*?(https?:\/\/[^\s"']+)/i);
                if (urlMatch) return urlMatch[1];
                
                return null;
            }''')
            
            # Try to find any JSON data embedded in the page
            embedded_data = await page.evaluate('''() => {
                // Look for script tags with JSON data
                const scripts = document.querySelectorAll('script[type="application/json"]');
                const data = [];
                scripts.forEach(s => {
                    try {
                        data.push(JSON.parse(s.textContent));
                    } catch (e) {}
                });
                return data;
            }''')
            
            return {
                'url': url,
                'page_text': page_text,
                'page_html': page_html,
                'file_links': all_file_links,
                'submit_url': submit_url,
                'embedded_data': embedded_data,
            }
            
        finally:
            await page.close()
            
    async def download_file(self, url: str, save_dir: Optional[str] = None) -> Tuple[str, bytes]:
        """
        Download a file from a URL.
        
        Args:
            url: The file URL to download.
            save_dir: Optional directory to save the file.
            
        Returns:
            Tuple of (filename, file_content_bytes).
        """
        # Use httpx for file downloads (faster than browser downloads)
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Extract filename from URL or Content-Disposition header
            filename = urlparse(url).path.split('/')[-1]
            if 'content-disposition' in response.headers:
                cd = response.headers['content-disposition']
                if 'filename=' in cd:
                    filename = cd.split('filename=')[-1].strip('"\'')
            
            content = response.content
            
            # Optionally save to disk
            if save_dir:
                filepath = os.path.join(save_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(content)
                logger.info(f"Downloaded file saved to {filepath}")
                
            return filename, content
            
    async def download_files_from_page(self, file_links: List[Dict], save_dir: str) -> List[Dict]:
        """
        Download multiple files from extracted links.
        
        Args:
            file_links: List of file link dictionaries from extract_quiz_data.
            save_dir: Directory to save downloaded files.
            
        Returns:
            List of dictionaries with download info and local paths.
        """
        downloaded = []
        
        for link in file_links:
            try:
                filename, content = await self.download_file(link['url'], save_dir)
                downloaded.append({
                    'original_url': link['url'],
                    'filename': filename,
                    'local_path': os.path.join(save_dir, filename),
                    'size': len(content),
                    'text': link.get('text', ''),
                })
                logger.info(f"Downloaded: {filename} ({len(content)} bytes)")
            except Exception as e:
                logger.error(f"Failed to download {link['url']}: {e}")
                downloaded.append({
                    'original_url': link['url'],
                    'error': str(e),
                })
                
        return downloaded


# Synchronous wrapper functions for non-async contexts
def fetch_page_content(url: str) -> Tuple[str, str]:
    """
    Synchronous wrapper to fetch page content.
    
    Args:
        url: URL to fetch.
        
    Returns:
        Tuple of (HTML content, text content).
    """
    async def _fetch():
        async with BrowserService() as browser:
            return await browser.get_page_content(url)
    
    return asyncio.run(_fetch())


def extract_quiz_info(url: str) -> Dict[str, Any]:
    """
    Synchronous wrapper to extract quiz data.
    
    Args:
        url: Quiz URL.
        
    Returns:
        Dictionary with quiz data.
    """
    async def _extract():
        async with BrowserService() as browser:
            return await browser.extract_quiz_data(url)
    
    return asyncio.run(_extract())
