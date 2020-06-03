---
layout: home
title: Yacine Haddad
image: about.jpg
---

<ul class="timeline">
  {% for event in site.data.timeline %}
  <li>
    <p class="timeline-date"> {{ event.date }}</p>
    <div class="timeline-content">
      <h3>{{ event.title }}</h3>
      <p>{{ event.content }}</p>
      <em style="color:gray;font-size:16px;margin-top:0px">
        {{ event.loc }}
      </em>
    </div>
  </li>
  {% endfor %}
</ul>
