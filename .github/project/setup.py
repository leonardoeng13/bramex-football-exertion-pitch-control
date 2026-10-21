# Cria o GitHub Project da equipe BraMex (campos, itens, README). Execução única.
import json, os, sys, datetime as dt, urllib.request

TOKEN = os.environ["PROJECT_TOKEN"]
OWNER, REPO = "leonardoeng13", "bramex-football-exertion-pitch-control"
TITLE = "BraMex · Team plan"
HERE = os.path.dirname(__file__)
# Lab semanal às terças; Week 6 = 20/out/2026
DUE = {f"W{w}": dt.date(2026, 10, 20) + dt.timedelta(weeks=w - 6) for w in range(6, 13)}

def gql(q, **v):
    req = urllib.request.Request("https://api.github.com/graphql", json.dumps({"query": q, "variables": v}).encode(),
                                 {"Authorization": f"bearer {TOKEN}", "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    if r.get("errors"):
        raise RuntimeError(json.dumps(r["errors"]))
    return r["data"]

d = gql("query($o:String!,$r:String!){viewer{id projectsV2(first:50){nodes{title}}} repository(owner:$o,name:$r){id}}", o=OWNER, r=REPO)
if any(n["title"] == TITLE for n in d["viewer"]["projectsV2"]["nodes"]):
    sys.exit(f"Project '{TITLE}' already exists; nothing done.")
pid = gql("mutation($o:ID!,$t:String!,$r:ID!){createProjectV2(input:{ownerId:$o,title:$t,repositoryId:$r}){projectV2{id number url}}}",
          o=d["viewer"]["id"], t=TITLE, r=d["repository"]["id"])["createProjectV2"]["projectV2"]
P = pid["id"]; print("created", pid["url"])

readme = open(os.path.join(HERE, "README_project.md"), encoding="utf-8").read()
gql("mutation($p:ID!,$s:String!,$m:String!){updateProjectV2(input:{projectId:$p,shortDescription:$s,readme:$m,public:false}){projectV2{id}}}",
    p=P, s="Plan, schedule, decisions and contributions for team BraMex (MITxT 2.980x).", m=readme)

def opts(names, colors):
    return [{"name": n, "color": c, "description": ""} for n, c in zip(names, colors)]

# Status: substitui as opções padrão
fields = gql("query($p:ID!){node(id:$p){...on ProjectV2{fields(first:30){nodes{...on ProjectV2SingleSelectField{id name}}}}}}", p=P)["node"]["fields"]["nodes"]
status_id = next(f["id"] for f in fields if f.get("name") == "Status")
STATUS = ["Not started", "In progress", "In review", "Blocked", "Done"]
START = "Not started"
try:
    gql("mutation($f:ID!,$o:[ProjectV2SingleSelectFieldOptionInput!]){updateProjectV2Field(input:{fieldId:$f,singleSelectOptions:$o}){projectV2Field{...on ProjectV2SingleSelectField{id}}}}",
        f=status_id, o=opts(STATUS, ["GRAY", "YELLOW", "BLUE", "RED", "GREEN"]))
except Exception as e:
    # API sem suporte a editar opções: mantém Todo / In Progress / Done
    print("status options not updated:", e); START = "Todo"

WS = ["Management", "Shared pipeline", "A · Players & load", "B · Teams", "C · Chances & possession changes", "Prediction", "Deliverables"]
SPEC = [
    ("Workstream", "SINGLE_SELECT", opts(WS, ["GRAY", "PURPLE", "ORANGE", "BLUE", "GREEN", "PINK", "RED"])),
    ("Owner", "SINGLE_SELECT", opts(["Leonardo", "Monica", "Pablo", "All"], ["ORANGE", "BLUE", "GREEN", "GRAY"])),
    ("Support", "TEXT", None),
    ("Type", "SINGLE_SELECT", opts(["Task", "Deliverable", "Milestone", "Decision"], ["GRAY", "RED", "PURPLE", "YELLOW"])),
    ("Week", "SINGLE_SELECT", opts(list(DUE), ["GRAY"] * len(DUE))),
    ("Start date", "DATE", None), ("Target date", "DATE", None),
    ("Task weight", "NUMBER", None),
    ("Role · Leonardo", "NUMBER", None), ("Role · Monica", "NUMBER", None), ("Role · Pablo", "NUMBER", None),
    ("Options compared", "TEXT", None), ("Agreed option", "TEXT", None),
]
F = {}
for name, typ, o in SPEC:
    q = "mutation($p:ID!,$n:String!,$t:ProjectV2CustomFieldType!" + (",$o:[ProjectV2SingleSelectFieldOptionInput!]" if o else "") + \
        "){createProjectV2Field(input:{projectId:$p,dataType:$t,name:$n" + (",singleSelectOptions:$o" if o else "") + \
        "}){projectV2Field{...on ProjectV2Field{id} ...on ProjectV2SingleSelectField{id options{id name}}}}}"
    r = gql(q, p=P, n=name, t=typ, **({"o": o} if o else {}))["createProjectV2Field"]["projectV2Field"]
    F[name] = r
st = gql("query($p:ID!){node(id:$p){...on ProjectV2{field(name:\"Status\"){...on ProjectV2SingleSelectField{id options{id name}}}}}}", p=P)["node"]["field"]
F["Status"] = st

def setv(item, name, val):
    if val is None or val == "":
        return
    f = F[name]
    if "options" in f:
        v = {"singleSelectOptionId": next(o["id"] for o in f["options"] if o["name"] == val)}
    elif isinstance(val, dt.date):
        v = {"date": val.isoformat()}
    elif isinstance(val, (int, float)):
        v = {"number": float(val)}
    else:
        v = {"text": val}
    gql("mutation($p:ID!,$i:ID!,$f:ID!,$v:ProjectV2FieldValue!){updateProjectV2ItemFieldValue(input:{projectId:$p,itemId:$i,fieldId:$f,value:$v}){projectV2Item{id}}}",
        p=P, i=item, f=f["id"], v=v)

def add(title, body):
    return gql("mutation($p:ID!,$t:String!,$b:String){addProjectV2DraftIssue(input:{projectId:$p,title:$t,body:$b}){projectItem{id}}}",
               p=P, t=title, b=body)["addProjectV2DraftIssue"]["projectItem"]["id"]

for wk, title, ws, owner, support, typ, w, roles, note in json.load(open(os.path.join(HERE, "items.json"), encoding="utf-8")):
    i = add(title, note)
    due = DUE[wk]
    for k, v in [("Status", START), ("Workstream", ws), ("Owner", owner), ("Support", support), ("Type", typ),
                 ("Week", wk), ("Start date", due - dt.timedelta(days=6)), ("Target date", due), ("Task weight", w)]:
        setv(i, k, v)
    if roles:
        for who, r in zip(["Leonardo", "Monica", "Pablo"], roles):
            setv(i, f"Role · {who}", r)
    print("item", wk, title)

for title, options, ws, owner in json.load(open(os.path.join(HERE, "decisions.json"), encoding="utf-8")):
    i = add(f"Decision: {title}", f"Options to compare: {options}\n\nRecord the agreed option and the reason in the fields and here.")
    for k, v in [("Status", START), ("Workstream", ws), ("Owner", owner), ("Type", "Decision"), ("Week", "W7"),
                 ("Start date", DUE["W6"]), ("Target date", DUE["W7"]), ("Options compared", options)]:
        setv(i, k, v)
    print("decision", title)
print("done", pid["url"])
