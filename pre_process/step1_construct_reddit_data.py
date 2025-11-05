import json
import multiprocessing


def load_submissions(filename):
    submissions = {}
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            data['comments'] = []  # 初始化评论列表
            submissions[data['id']] = data

            if "CMV: Society is apathetic to child abuse and it's victims" in data["title"]:
                print(data["id"])
                print(data)
    return submissions

submission_file = "/home/sunyang/hlt/reddit/subreddits23/changemyview_submissions.jsonl"
submissions = load_submissions(submission_file)

comment_file = "/home/sunyang/hlt/reddit/subreddits23/changemyview_comments.jsonl"
with open(comment_file, 'r') as file:
    comments = []
    for line in file:
        data = json.loads(line)
        comments.append(data)


results = {}
for data in comments:
    link_id = data['link_id'][3:]  # Remove 't3_' prefix
    if link_id in submissions:
        if link_id not in results:
            results[link_id] = []
        results[link_id].append(data)


for link_id in results.keys():
    submissions[link_id]['comments'].extend(results[link_id])


# Sort comments by 'created_utc' for each submission
for post_id in submissions.keys():
    submissions[post_id]['comments'].sort(key=lambda x: int(x['created_utc']))


# Output or save your data here
# Example: print(submissions['some_post_id'])
# with open("/home/sunyang/hlt/reddit/subreddits23/changemyview_datas.jsonl", 'w', encoding='utf-8') as f:
#     for i, post in submissions.items():
#         f.write(json.dumps(post, ensure_ascii=False) + '\n')