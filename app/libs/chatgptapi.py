import requests
import json
from logzero import logger
from datetime import datetime, timedelta


def check_credit_usage_by_date(api_token):
    date_format = '%Y-%m-%d'
    end_date = datetime.utcnow().strftime(date_format)
    start_date = (datetime.utcnow() - timedelta(days=100)
                  ).strftime(date_format)

    url = f"https://api.openai.com/dashboard/billing/usage?start_date={start_date}&end_date={end_date}"

    payload = {}
    headers = {
        'Authorization': f'Bearer {api_token}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    logger.info(response.status_code)
    data = response.json()
    if response.status_code == 200 and data:
        total_usage = data.get('total_usage')
        if total_usage:
            total_usage = float(total_usage) / 100
            logger.info(total_usage)
    else:
        logger.info(response.text)


def get_credit_usage(api_token):
    url = "https://api.openai.com/dashboard/billing/subscription"

    payload = {}
    headers = {
        'Authorization': f'Bearer {api_token}'
    }

    logger.info(headers)
    response = requests.request("GET", url, headers=headers, data=payload)
    logger.info(response.status_code)
    data = response.json()
    if response.status_code == 200 and data:
        soft_limit_usd = data.get('soft_limit_usd')
        hard_limit_usd = data.get('hard_limit_usd')
        if soft_limit_usd and hard_limit_usd:
            fund_remain = float(hard_limit_usd) - float(soft_limit_usd)
            logger.info(fund_remain)
    else:
        logger.info(response.text)


def main():
    api_token = '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'
    get_credit_usage(api_token)


if __name__ == '__main__':
    main()
