"""
Specialized question handlers for various quiz types.

This module provides handlers for different types of data analysis questions:
- DOM parsing (hidden elements, reversed text)
- API pagination
- API authentication
- Data cleaning
- CSV processing
- JavaScript execution
- Date manipulation
- Spatial analysis
- Log parsing (regex)
- Data pipeline (joins)
"""

import asyncio
import base64
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QuestionHandlers:
    """Specialized handlers for different question types."""
    
    def __init__(self, llm_helper=None):
        """Initialize with optional LLM helper."""
        self.llm = llm_helper
        self.http_client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        return self.http_client
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    # ========================================
    # DOM PARSING
    # ========================================
    
    def handle_hidden_element(self, html: str, class_name: str, reverse: bool = False) -> Optional[str]:
        """
        Extract text from a hidden element by class name.
        
        Args:
            html: Full HTML content.
            class_name: CSS class name to find.
            reverse: Whether to reverse the text.
            
        Returns:
            Extracted (and optionally reversed) text.
        """
        # Try multiple regex patterns
        patterns = [
            rf'class=["\'](?:[^"\']*\s)?{class_name}(?:\s[^"\']*)?["\'][^>]*>([^<]+)<',
            rf'class="{class_name}"[^>]*>([^<]+)<',
            rf"class='{class_name}'[^>]*>([^<]+)<",
            rf'class={class_name}[^>]*>([^<]+)<',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                text = match.group(1).strip()
                if reverse:
                    text = text[::-1]
                logger.info(f"Found hidden element '{class_name}': {text}")
                return text
        
        # Try finding by data attribute
        data_pattern = rf'data-class=["\']?{class_name}["\']?[^>]*>([^<]+)<'
        match = re.search(data_pattern, html, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            if reverse:
                text = text[::-1]
            return text
        
        logger.warning(f"Could not find element with class '{class_name}'")
        return None
    
    # ========================================
    # API PAGINATION
    # ========================================
    
    async def handle_pagination(
        self, 
        base_url: str, 
        target_id: int,
        id_field: str = "id",
        name_field: str = "name"
    ) -> Optional[str]:
        """
        Traverse paginated API to find item by ID.
        
        Args:
            base_url: Base URL with page parameter (e.g., .../items?page=1)
            target_id: The ID to search for.
            id_field: Field name for ID in response.
            name_field: Field name to return.
            
        Returns:
            The name/value of the item with target ID.
        """
        client = await self._get_client()
        page = 1
        max_pages = 100  # Safety limit
        
        while page <= max_pages:
            # Handle URL with or without existing query params
            if '?' in base_url:
                url = re.sub(r'page=\d+', f'page={page}', base_url)
                if 'page=' not in url:
                    url = f"{url}&page={page}"
            else:
                url = f"{base_url}?page={page}"
            
            try:
                response = await client.get(url)
                data = response.json()
                
                # Handle different response formats
                items = data if isinstance(data, list) else data.get('items', data.get('data', []))
                
                if not items:
                    logger.info(f"Empty page at {page}, stopping pagination")
                    break
                
                # Search for target ID
                for item in items:
                    item_id = item.get(id_field)
                    if item_id == target_id or str(item_id) == str(target_id):
                        result = item.get(name_field)
                        logger.info(f"Found item {target_id}: {result}")
                        return result
                
                page += 1
                
            except Exception as e:
                logger.error(f"Pagination error at page {page}: {e}")
                break
        
        logger.warning(f"Item with ID {target_id} not found")
        return None
    
    # ========================================
    # API AUTHENTICATION
    # ========================================
    
    async def fetch_with_auth(
        self, 
        url: str, 
        headers: Dict[str, str] = None,
        api_key: str = None,
        api_key_header: str = "X-API-Key"
    ) -> Any:
        """
        Fetch data from authenticated API endpoint.
        
        Args:
            url: API endpoint URL.
            headers: Custom headers dict.
            api_key: API key value.
            api_key_header: Header name for API key.
            
        Returns:
            Parsed JSON response.
        """
        client = await self._get_client()
        
        req_headers = headers or {}
        if api_key:
            req_headers[api_key_header] = api_key
        
        try:
            response = await client.get(url, headers=req_headers)
            return response.json()
        except Exception as e:
            logger.error(f"Auth API fetch failed: {e}")
            return None
    
    def find_max_by_field(self, data: List[Dict], value_field: str, return_field: str) -> Any:
        """Find item with maximum value in a field."""
        if not data:
            return None
            
        # Filter out non-dict items to avoid AttributeError
        valid_items = [item for item in data if isinstance(item, dict)]
        if not valid_items:
            return None
        
        max_item = max(valid_items, key=lambda x: float(x.get(value_field, 0) or 0))
        return max_item.get(return_field)
    
    # ========================================
    # DATA CLEANING
    # ========================================
    
    def clean_price(self, value: Any) -> Optional[float]:
        """
        Clean a messy price value to extract numeric amount.
        
        Handles: "$123.45", "123,456.78", "Rs. 100", nulls, strings, etc.
        """
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove currency symbols and whitespace
            cleaned = re.sub(r'[^\d.,\-]', '', value)
            if not cleaned:
                return None
            
            # Handle different number formats
            # Remove thousand separators (commas if followed by 3 digits)
            cleaned = re.sub(r',(\d{3})', r'\1', cleaned)
            # Also handle European format (dots as thousands)
            if cleaned.count('.') > 1:
                cleaned = cleaned.replace('.', '')
            cleaned = cleaned.replace(',', '.')
            
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    async def handle_dirty_data(self, url: str, price_field: str = "price") -> float:
        """
        Fetch and clean dirty data, sum valid prices.
        
        Args:
            url: API endpoint with dirty data.
            price_field: Field name containing prices.
            
        Returns:
            Sum of all valid numeric prices.
        """
        client = await self._get_client()
        
        try:
            response = await client.get(url)
            data = response.json()
            
            items = data if isinstance(data, list) else data.get('items', data.get('data', []))
            
            total = 0.0
            valid_count = 0
            
            for item in items:
                price = self.clean_price(item.get(price_field))
                if price is not None:
                    total += price
                    valid_count += 1
            
            logger.info(f"Cleaned {valid_count} valid prices, total: {total}")
            return total
            
        except Exception as e:
            logger.error(f"Dirty data handling failed: {e}")
            return 0.0
    
    # ========================================
    # CSV PROCESSING
    # ========================================
    
    async def handle_csv_analysis(
        self,
        url: str,
        filters: Dict[str, Any] = None,
        sum_column: str = None,
        count: bool = False
    ) -> Any:
        """
        Download and analyze CSV with filters.
        
        Args:
            url: CSV file URL.
            filters: Dict of column->value filters.
            sum_column: Column to sum.
            count: If True, return count instead of sum.
            
        Returns:
            Computed result.
        """
        client = await self._get_client()
        
        try:
            response = await client.get(url)
            df = pd.read_csv(pd.io.common.BytesIO(response.content))
            
            logger.info(f"CSV loaded: {df.shape}, columns: {list(df.columns)}")
            
            # Apply filters
            if filters:
                for col, val in filters.items():
                    if col in df.columns:
                        df = df[df[col] == val]
                        logger.info(f"Filtered {col}={val}, rows: {len(df)}")
            
            if count:
                return len(df)
            
            if sum_column and sum_column in df.columns:
                result = df[sum_column].sum()
                logger.info(f"Sum of {sum_column}: {result}")
                return result
            
            # Return as list of dicts for JSON serialization
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"CSV analysis failed: {e}")
            return None
    
    # ========================================
    # JAVASCRIPT EXECUTION
    # ========================================
    
    async def handle_js_execution(self, html: str, page_text: str) -> Optional[str]:
        """
        Extract result from JavaScript execution.
        
        The browser already executes JS, so we look for the rendered result.
        Also try to find and evaluate simple JS patterns.
        
        Args:
            html: Full HTML content.
            page_text: Rendered page text.
            
        Returns:
            The computed result.
        """
        # Look for common result patterns in rendered text
        patterns = [
            r'(?:result|answer|output|value)[:\s]+(\d+)',
            r'(?:The (?:number|answer|result) is)[:\s]+(\d+)',
            r'(?:generated|calculated)[:\s]+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Try to find and evaluate simple JS
        # Look for script content
        script_match = re.search(r'<script[^>]*>(.*?)</script>', html, re.DOTALL | re.IGNORECASE)
        if script_match:
            script = script_match.group(1)
            
            # Look for simple calculations
            calc_match = re.search(r'(\d+)\s*[\+\-\*\/]\s*(\d+)', script)
            if calc_match:
                try:
                    result = eval(calc_match.group(0))
                    return str(int(result))
                except:
                    pass
            
            # Look for variable assignments with numbers
            var_match = re.search(r'(?:let|var|const)\s+\w+\s*=\s*(\d+)', script)
            if var_match:
                return var_match.group(1)
        
        # If LLM available, ask it to analyze
        if self.llm and self.llm.is_available():
            try:
                answer = await self.llm.solve_question_directly(
                    f"What number is generated by this JavaScript?\n\n{html[:3000]}",
                    None
                )
                # Extract number from answer
                num_match = re.search(r'\d+', answer)
                if num_match:
                    return num_match.group(0)
            except:
                pass
        
        return None
    
    # ========================================
    # DATE MANIPULATION
    # ========================================
    
    async def handle_date_analysis(
        self,
        url: str,
        target_weekday: int = None,  # 0=Monday, 1=Tuesday, etc.
        date_field: str = None
    ) -> int:
        """
        Fetch dates and count by weekday.
        
        Args:
            url: API endpoint with dates.
            target_weekday: Weekday to count (0-6, Monday-Sunday).
            date_field: Field name if dates are in objects.
            
        Returns:
            Count of dates matching criteria.
        """
        client = await self._get_client()
        
        try:
            response = await client.get(url)
            data = response.json()
            
            # Handle different formats
            if isinstance(data, list):
                if data and isinstance(data[0], str):
                    dates = data
                elif data and isinstance(data[0], dict) and date_field:
                    dates = [item.get(date_field) for item in data]
                else:
                    dates = data
            else:
                dates = data.get('dates', data.get('timestamps', []))
            
            count = 0
            for date_str in dates:
                if not date_str:
                    continue
                try:
                    # Parse ISO 8601
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if target_weekday is not None and dt.weekday() == target_weekday:
                        count += 1
                except Exception as e:
                    logger.debug(f"Could not parse date {date_str}: {e}")
            
            logger.info(f"Found {count} dates on weekday {target_weekday}")
            return count
            
        except Exception as e:
            logger.error(f"Date analysis failed: {e}")
            return 0
    
    # ========================================
    # SPATIAL ANALYSIS
    # ========================================
    
    async def handle_spatial_distance(
        self,
        url: str,
        point_a_id: str,
        point_b_id: str,
        decimal_places: int = 2
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            url: API endpoint with location data.
            point_a_id: ID of first point.
            point_b_id: ID of second point.
            decimal_places: Decimal places for rounding.
            
        Returns:
            Euclidean distance rounded to specified places.
        """
        client = await self._get_client()
        
        try:
            response = await client.get(url)
            data = response.json()
            
            points = data if isinstance(data, list) else data.get('points', data.get('locations', []))
            
            point_a = None
            point_b = None
            
            for point in points:
                pid = point.get('id', point.get('ID'))
                if pid == point_a_id:
                    point_a = point
                elif pid == point_b_id:
                    point_b = point
            
            if not point_a or not point_b:
                logger.warning(f"Could not find points {point_a_id} and/or {point_b_id}")
                return 0.0
            
            # Get coordinates
            x1 = point_a.get('x', point_a.get('coordinates', [0, 0])[0])
            y1 = point_a.get('y', point_a.get('coordinates', [0, 0])[1])
            x2 = point_b.get('x', point_b.get('coordinates', [0, 0])[0])
            y2 = point_b.get('y', point_b.get('coordinates', [0, 0])[1])
            
            # Handle [x, y] format
            if isinstance(x1, list):
                x1, y1 = x1[0], x1[1]
            if isinstance(x2, list):
                x2, y2 = x2[0], x2[1]
            
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            result = round(distance, decimal_places)
            
            logger.info(f"Distance from {point_a_id} to {point_b_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Spatial analysis failed: {e}")
            return 0.0
    
    # ========================================
    # LOG PARSING
    # ========================================
    
    async def handle_log_parsing(
        self,
        url: str,
        find: str = "most_common_ip"
    ) -> str:
        """
        Parse server logs and extract information.
        
        Args:
            url: API endpoint with log data.
            find: What to find ("most_common_ip", etc.)
            
        Returns:
            Extracted value.
        """
        client = await self._get_client()
        
        try:
            response = await client.get(url)
            
            # Handle different response types
            content_type = response.headers.get('content-type', '')
            if 'json' in content_type:
                data = response.json()
                logs = data if isinstance(data, list) else data.get('logs', [])
                log_text = '\n'.join(str(log) for log in logs)
            else:
                log_text = response.text
            
            if find == "most_common_ip":
                # Extract all IPs using regex
                ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
                ips = re.findall(ip_pattern, log_text)
                
                if ips:
                    counter = Counter(ips)
                    most_common = counter.most_common(1)[0][0]
                    logger.info(f"Most common IP: {most_common} ({counter[most_common]} occurrences)")
                    return most_common
            
            return None
            
        except Exception as e:
            logger.error(f"Log parsing failed: {e}")
            return None
    
    # ========================================
    # DATA PIPELINE (JOINS)
    # ========================================
    
    async def handle_data_pipeline(
        self,
        endpoints: Dict[str, str],
        join_logic: str = None
    ) -> Any:
        """
        Fetch and join data from multiple endpoints.
        
        Args:
            endpoints: Dict of name->URL for each endpoint.
            join_logic: Description of join/filter logic.
            
        Returns:
            Computed result based on logic.
        """
        client = await self._get_client()
        
        data = {}
        for name, url in endpoints.items():
            try:
                response = await client.get(url)
                data[name] = response.json()
                logger.info(f"Fetched {name}: {len(data[name]) if isinstance(data[name], list) else 'object'}")
            except Exception as e:
                logger.error(f"Failed to fetch {name}: {e}")
                data[name] = []
        
        # Common pattern: users, products, orders join
        if 'users' in data and 'products' in data and 'orders' in data:
            return self._join_users_products_orders(
                data['users'], 
                data['products'], 
                data['orders'],
                user_tier='gold'
            )
        
        return data
    
    def _join_users_products_orders(
        self,
        users: List[Dict],
        products: List[Dict],
        orders: List[Dict],
        user_tier: str = None
    ) -> float:
        """
        Join users, products, and orders data.
        
        Logic:
        1. Filter users by tier
        2. Find orders for those users
        3. Calculate total value from products
        """
        # Validate inputs are lists of dicts
        if not isinstance(users, list) or not isinstance(products, list) or not isinstance(orders, list):
            logger.warning("Data pipeline inputs must be lists")
            return 0.0
            
        # Filter users by tier
        if user_tier:
            filtered_users = [u for u in users if isinstance(u, dict) and u.get('tier', '').lower() == user_tier.lower()]
        else:
            filtered_users = [u for u in users if isinstance(u, dict)]
        
        user_ids = {u.get('id', u.get('user_id')) for u in filtered_users}
        logger.info(f"Filtered to {len(user_ids)} users with tier '{user_tier}'")
        
        # Create product price lookup
        product_prices = {}
        for p in products:
            if not isinstance(p, dict):
                continue
            pid = p.get('id', p.get('product_id'))
            price = p.get('price', 0)
            product_prices[pid] = price
        
        # Calculate total value of orders for filtered users
        total = 0.0
        for order in orders:
            if not isinstance(order, dict):
                continue
            order_user = order.get('user_id', order.get('userId'))
            if order_user in user_ids:
                # Get product(s) in order
                product_id = order.get('product_id', order.get('productId'))
                products_list = order.get('products', [product_id] if product_id else [])
                
                if not isinstance(products_list, list):
                    products_list = [products_list]
                
                for pid in products_list:
                    price = product_prices.get(pid, 0)
                    quantity = order.get('quantity', 1)
                    total += price * quantity
        
        logger.info(f"Total order value for {user_tier} users: {total}")
        return total


# Convenience instance
handlers = QuestionHandlers()
