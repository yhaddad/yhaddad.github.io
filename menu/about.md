---
layout: home
title: Yacine Haddad
image: about.jpg
---
I was born in 1985 in Algiers, Algeria. I studied physics in Algiers and Paris (France), and earned my Master’s Diploma in 2012.
After my bachelor’s degree in theoretical physics, I decided that particle physics was what I wanted to do. I started in this field working on my PhD thesis, building and studying the new generation of highly granular calorimeters within the CALICE collaboration, which are specially designed for the future linear collider machines (ILC). I was based at Ecole Polytechnique in Paris with various trips to CERN to participate in several beam tests. During my thesis, I investigated different algorithms to reconstruct particle showers and characterise this new kind of detector using the high granularity potential. I also explored a new method to measure the Higgs production in such a machine using the hadronic Z decay in the ZH production process. I joined the HiggsTools program in IPPP Durham and Imperial College London where I started to be involved in the CMS experiment.


### Timeline
<ul class="timeline">
  {% for event in site.data.timeline %}
  <li>
    <p class="timeline-date"> {{ event.date }}</p>
    <div class="timeline-content">
      <h3 class="timeline-h3">{{ event.title }}</h3>
      <p class="timeline-p">{{ event.content }}</p>
      <em style="color:gray;font-size:16px;margin-top:0px">
        {{ event.loc }}
      </em>
    </div>
  </li>
  {% endfor %}
</ul>
