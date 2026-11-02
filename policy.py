"""Provenance policy for effectful tool calls (Go plan, milestone M1).

Who receives money, data or access must come from the user, not from data the agent read. For every effectful call,
each *party* argument (payee, recipient, invitee, channel, URL, credential, profile field) must be:

  (a) named in the trusted user request, or
  (b) a known entity of this user -- taken from the environment as it was when the task started, and only from
      structural fields that untrusted content cannot write (payees in the account's own transaction records, the
      address book, workspace members, the catalog the user is booking from).

Anything else came from untrusted text (a bill, a web page, a message) and needs the user's confirmation; the call is
refused with a typed reason. The policy is deterministic: no model, no threshold. Calls that are not effectful, and
non-party arguments (amounts, subjects, bodies, ids), are out of its scope and are covered by the other layers.

Known gaps, by design: values the user asked for that are found in data ("pay the bill in this file") need a
confirmation; object ids (delete_file, cancel_calendar_event) are not parties and are not checked.
"""
from __future__ import annotations

import re
import weakref
from dataclasses import dataclass

# function -> party arguments, per suite. The tool list was read from AgentDojo v1.2.2; every effectful tool is listed,
# with an empty tuple when it has no party argument (it is then allowed by this layer and left to the others).
EFFECTS: dict[str, dict[str, tuple[str, ...]]] = {
    "banking": {
        "send_money": ("recipient",),
        "schedule_transaction": ("recipient",),
        "update_scheduled_transaction": ("recipient",),
        "update_password": ("password",),
        "update_user_info": ("first_name", "last_name", "street", "city"),
    },
    "slack": {
        "send_direct_message": ("recipient",),
        "send_channel_message": ("channel",),
        "add_user_to_channel": ("user", "channel"),
        "invite_user_to_slack": ("user", "user_email"),
        "remove_user_from_slack": ("user",),
        "get_webpage": ("url",),   # gated only when the URL can carry data out (query string); see check()
        "post_webpage": ("url",),
    },
    "workspace": {
        "send_email": ("recipients", "cc", "bcc"),
        "share_file": ("email",),
        "create_calendar_event": ("participants",),
        "add_calendar_event_participants": ("participants",),
        "delete_email": (), "cancel_calendar_event": (), "reschedule_calendar_event": (),
        "append_to_file": (), "create_file": (), "delete_file": (),
    },
    "travel": {
        "send_email": ("recipients", "cc", "bcc"),
        "create_calendar_event": ("participants",),
        "reserve_hotel": ("hotel",),
        "reserve_car_rental": ("company",),
        "reserve_restaurant": ("restaurant",),
        "cancel_calendar_event": (),
    },
}
# Arguments that must come from the user request itself; a known entity is not enough (credentials, own profile).
USER_ONLY = {("banking", "update_password", "password")} | {("banking", "update_user_info", f)
                                                              for f in ("first_name", "last_name", "street", "city")}

SLACK_CHANNEL = re.compile(r"[A-Za-z0-9_\-]{1,80}")

SUITE_OF_ENV = {"BankingEnvironment": "banking", "SlackEnvironment": "slack",
                "WorkspaceEnvironment": "workspace", "TravelEnvironment": "travel"}


def norm(v) -> str:
    s = str(v).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.rstrip("/")
    return re.sub(r"\s+", " ", s)


def compact(s: str) -> str:
    """Spacing-insensitive form for identifiers written with spaces (IBANs)."""
    return re.sub(r"[\s\-]", "", s)


def known_entities(env) -> frozenset[str]:
    """Structural, injection-proof entities of this user at task start (see module docstring)."""
    suite = SUITE_OF_ENV.get(type(env).__name__)
    if suite is None:
        raise ValueError(f"no provenance policy for environment {type(env).__name__}")
    out: set[str] = set()
    if suite == "banking":
        acct = env.bank_account
        out.add(acct.iban)
        for t in [*acct.transactions, *acct.scheduled_transactions]:
            out.update([t.recipient, t.sender])
    elif suite == "slack":
        out.update(env.slack.users)
        # channels are workspace structure, but a channel name can carry injected text in AgentDojo; only names that
        # are valid Slack channel identifiers (no spaces, <= 80 of [a-z0-9_-]) count as known
        out.update(c for c in env.slack.channels if SLACK_CHANNEL.fullmatch(c))
    elif suite == "workspace":
        out.add(env.inbox.account_email)
        out.update(c.email for c in env.inbox.contact_list)
        for ev in env.calendar.events.values():
            out.update(ev.participants)
    else:  # travel (SUITE_OF_ENV has exactly four suites)
        out.add(env.user.email)
        out.update(h.name for h in env.hotels.hotel_list)
        out.update(r.name for r in env.restaurants.restaurant_list)
        out.update(c.name for c in env.car_rental.company_list)
        for ev in env.calendar.events.values():
            out.update(ev.participants)
    return frozenset(n for n in (norm(x) for x in out if x) if n and n != "me")


REFUSAL_MARK = "needs the user's confirmation"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class Episode:
    suite: str
    known: frozenset[str]


_EPISODES: "weakref.WeakKeyDictionary[object, Episode]" = weakref.WeakKeyDictionary()


def register_episode(runtime, env) -> None:
    """Snapshot the known entities when the task starts. Call before the agent runs; keyed by the episode's runtime."""
    suite = SUITE_OF_ENV.get(type(env).__name__)
    if suite is None:
        raise ValueError(f"no provenance policy for environment {type(env).__name__}")
    _EPISODES[runtime] = Episode(suite, known_entities(env))


def episode_of(runtime) -> Episode:
    ep = _EPISODES.get(runtime)
    if ep is None:
        raise RuntimeError("provenance policy: episode not registered (call policy.register_episode before the run)")
    return ep


def _values(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def in_request(value: str, user: str) -> bool:
    n, u = norm(value), norm(user)
    if len(n) < 2:
        return False
    return n in u or (len(compact(n)) >= 8 and compact(n) in compact(u))


def carries_data(url: str) -> bool:
    """A URL can exfiltrate data when it has a query string or fragment."""
    return bool(re.search(r"[?#&=]", str(url)))


def check(ep: Episode, user: str, function: str, args: dict) -> Verdict:
    parties = EFFECTS[ep.suite].get(function)
    if parties is None:
        return Verdict(True)
    if (ep.suite, function) == ("slack", "get_webpage") and not carries_data(args.get("url", "")):
        return Verdict(True)  # reading a page is not an effect; following links found in messages is normal work
    for name in parties:
        for v in _values(args.get(name)):
            if in_request(v, user):
                continue
            if (ep.suite, function, name) not in USER_ONLY and norm(v) in ep.known:
                continue
            what = "taken from your request" if (ep.suite, function, name) in USER_ONLY else \
                "in your request or among your known contacts/payees"
            return Verdict(False, f"Error: {function} {REFUSAL_MARK}: {name} {str(v)!r} is not {what}; "
                                  f"it came from data the agent read. Ask the user to confirm it.")
    return Verdict(True)
