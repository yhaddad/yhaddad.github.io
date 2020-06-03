---
layout: page
title: Talks
---

{% for talk in site.data.talks %}
<span class="row">
    <span class="talk-margin">
        <span class="post-date">
            <i class="fa fa-calendar" aria-hidden="true"></i> {{ talk.date | date: "%b, %Y" }}
        </span>
    </span>
    <span class="talk-content">
      <a href="{{ talk.pdf }}" style="font-weight:bold;margin-top:0px;">
        {{ talk.title }}
      </a> 
      <br />
      <a href="{{ talk.conf.link }}" style="color:auto;font-size:16px;margin-top:0px">
        {{ talk.conf.name }}
      </a>
      <br />
      <em style="color:gray;font-size:16px;margin-top:0px">
        {{ talk.conf.loc }}
      </em>
  </span>
</span>
{% endfor %}