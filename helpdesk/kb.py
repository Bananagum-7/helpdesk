"""Help-center articles and a small BM25 search over them."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

ARTICLES = [
    {"id": "KB-01", "title": "Return and refund policy",
     "body": "You can return most items within 30 days of delivery for a full refund to the original payment method. "
             "Gift cards and clearance items are final sale and cannot be refunded. Orders that have not shipped yet can be "
             "cancelled for a full refund. Refunds take 5-7 business days to appear on your statement."},
    {"id": "KB-02", "title": "Shipping times and carriers",
     "body": "Standard shipping takes 3-7 business days within India. We ship with BlueDart, Delhivery, DHL and UPS. "
             "You get a tracking number by email once your order ships. Express shipping (1-2 days) is available in metro cities."},
    {"id": "KB-03", "title": "Tracking your order",
     "body": "Open My Orders and select the order to see its status and tracking number. Tracking can take up to 24 hours to "
             "update after the order ships. If tracking hasn't moved in 5 days, contact support."},
    {"id": "KB-04", "title": "Resetting your password",
     "body": "Click 'Forgot password' on the sign-in page and enter your email. The reset link expires after 1 hour. "
             "If you don't get the email, check spam or ask support to resend it."},
    {"id": "KB-05", "title": "Two-factor authentication",
     "body": "Turn on two-factor authentication in Settings > Security. You can use an authenticator app or SMS. "
             "If you lose access to your device, use one of your backup codes or contact support with ID verification."},
    {"id": "KB-06", "title": "Payment methods",
     "body": "We accept credit and debit cards, UPI, net banking and gift cards. Cash on delivery is available for orders under 5000 rupees. "
             "We do not accept cryptocurrency."},
    {"id": "KB-07", "title": "Warranty",
     "body": "Electronics carry a 1-year manufacturer warranty. Office chairs have a 3-year warranty on the frame. "
             "Warranty claims need the order number and a photo or video of the fault."},
    {"id": "KB-08", "title": "Damaged or wrong item",
     "body": "If an item arrives damaged or you received the wrong item, contact us within 7 days of delivery with a photo. "
             "We'll send a replacement or refund at no cost, including return shipping."},
    {"id": "KB-09", "title": "Cancelling an order",
     "body": "You can cancel an order from My Orders until it ships. After it ships, wait for delivery and start a return instead."},
    {"id": "KB-10", "title": "International shipping",
     "body": "We currently ship only within India. International shipping is not available."},
    {"id": "KB-11", "title": "Gold membership",
     "body": "Gold members get free express shipping and priority support. Gold membership is by invitation based on purchase history."},
    {"id": "KB-12", "title": "Changing your delivery address",
     "body": "You can change the delivery address while the order is processing. Once shipped, the address can only be changed by the carrier."},
]

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the and or of to in on for is are i my me can do does how what when you your it with be".split())


def _tok(t: str) -> list[str]:
    return [w for w in _TOKEN.findall(t.lower()) if w not in _STOP]


_bm25 = BM25Okapi([_tok(a["title"] + " " + a["body"]) for a in ARTICLES])


def search(query: str, k: int = 3) -> list[dict]:
    scores = _bm25.get_scores(_tok(query))
    ranked = sorted(zip(ARTICLES, scores), key=lambda x: -x[1])
    return [{**a, "score": round(float(s), 2)} for a, s in ranked[:k] if s > 0]
